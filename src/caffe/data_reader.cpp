#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

using boost::weak_ptr;

map<const string, weak_ptr<DataReader::Body> > DataReader::bodies_;
static boost::mutex bodies_mutex_;

DataReader::DataReader(const LayerParameter& param)
    : queue_pair_(new QueuePair(  //
        param.data_param().prefetch() * param.data_param().batch_size())) {
  // Get or create a body
  boost::mutex::scoped_lock lock(bodies_mutex_);
  string key = source_key(param);
  weak_ptr<Body>& weak = bodies_[key];
  body_ = weak.lock();
  if (!body_) {
    body_.reset(new Body(param));
    bodies_[key] = weak_ptr<Body>(body_);
  }
  body_->new_queue_pairs_.push(queue_pair_);
}

DataReader::~DataReader() {
  string key = source_key(body_->param_);
  body_.reset();
  boost::mutex::scoped_lock lock(bodies_mutex_);
  if (bodies_[key].expired()) {
    bodies_.erase(key);
  }
}

//

DataReader::QueuePair::QueuePair(int size) {
  // Initialize the free queue with requested number of datums
  for (int i = 0; i < size; ++i) {
    free_.push(new Datum());
  }
}

DataReader::QueuePair::~QueuePair() {
  Datum* datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}

//

DataReader::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {
  StartInternalThread();
}

DataReader::Body::~Body() {
  StopInternalThread();
}

void DataReader::Body::InternalThreadEntry() {
  int num_sources = param_.data_param().source_size();
  int num_proportion = param_.data_param().proportion_size();

  if (num_sources > 1)
    CHECK_EQ(num_sources, num_proportion);

  vector<shared_ptr<db::DB> > dbs(num_sources);
  vector<shared_ptr<db::Cursor> > cursors(dbs.size());

  for (int i = 0; i < dbs.size(); ++i) {
    dbs[i] = shared_ptr<db::DB>(db::GetDB(param_.data_param().backend()));
    dbs[i]->Open(param_.data_param().source(i), db::READ);
    cursors[i] = shared_ptr<db::Cursor>(dbs[i]->NewCursor());
  }

  // Ex 1: proportion = [] => round_robin_mapping = [0]
  // Ex 2: proportion = [2, 3, 2] => round_robin_mapping = [0, 0, 1, 1, 1, 2, 2]
  vector<int> round_robin_mapping;

  if (num_sources == 1) {
    round_robin_mapping.push_back(0);
  }
  else {
    for (int i=0; i<num_proportion; ++i) {
      int p = param_.data_param().proportion(i);
      for (int j=0; j<p; ++j)
	round_robin_mapping.push_back(i);
    }
  }

  vector<shared_ptr<QueuePair> > qps;
  try {
    int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so read one item, then wait for the next solver.
    for (int i = 0; i < solver_count; ++i) {
      shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
      read_one(cursors[0].get(), qp.get());
      qps.push_back(qp);
    }
    // Main loop
    size_t counter = 0;
    while (!must_stop()) {
      int source_id = round_robin_mapping[++counter % round_robin_mapping.size()];
      // printf("Using \33[33m%s\33[0m\n", param_.data_param().source(source_id).c_str());

      for (int i = 0; i < solver_count; ++i) {
	read_one(cursors[source_id].get(), qps[i].get());
      }
      // Check no additional readers have been created. This can happen if
      // more than one net is trained at a time per process, whether single
      // or multi solver. It might also happen if two data layers have same
      // name and same source.
      CHECK_EQ(new_queue_pairs_.size(), 0);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

void DataReader::Body::read_one(db::Cursor* cursor, QueuePair* qp) {
  Datum* datum = qp->free_.pop();
  // TODO deserialize in-place instead of copy?
  datum->ParseFromString(cursor->value());
  qp->full_.push(datum);

  // go to the next iter
  cursor->Next();
  if (!cursor->valid()) {
    DLOG(INFO) << "Restarting data prefetching from start.";
    cursor->SeekToFirst();
  }
}

}  // namespace caffe
