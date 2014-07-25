from mrjob.job import MRJob
import sys, random, numpy, pickle


class UpdateCentroids(MRJob):

  def steps(self):
    return [self.mr(
      mapper_init = self.read_centroids,
      mapper      = self.find_closest_centorid,
      combiner    = self.partial_sum,
      reducer     = self.recalculate_centroids
      )
    ]

  def read_centroids(self):
    f = open(self.options.centroids)
    self.centroids = pickle.load(f)
    f.close()

  def convert_string_to_vector(self, _line):
    return [float(i) for i in _line.split('\n')[0].split(',')[1:]]

  def distance(self, _vector, _centroid):
    # import numpy
    vector   = numpy.array(_vector)
    centroid = numpy.array(_centroid)
    distance = numpy.linalg.norm(vector - centroid)
    return distance

  def find_closest_centorid(self, _, _line):
    vector = self.convert_string_to_vector(_line)
    if vector != []:
      centroid_distances = [self.distance(vector,centroid) for centroid in self.centroids]
      shortest_distance  = min(centroid_distances)
      closest_centroid   = [i for i, value in enumerate(centroid_distances) if value == shortest_distance][0]

      yield closest_centroid, vector

  def partial_sum(self, _cluster, _vectors):
    vector_sum = numpy.array(_vectors.next())
    vector_count = 1
 
    for vector in _vectors:
      vector_count += 1
      vector_sum += vector

    yield _cluster, (vector_sum.tolist(), vector_count)

  def recalculate_centroids(self, _cluster, _partial_sums):
    vector_total_sum, vector_total_count = _partial_sums.next()
    vector_total_sum = numpy.array(vector_total_sum)

    for vector_sum, vector_total_count in _partial_sums:
      vector_total_sum   += vector_sum
      vector_total_count += count

    yield (vector_total_sum / vector_total_count).tolist(), None

  def configure_options(self):
    super(UpdateCentroids, self).configure_options()
    self.add_passthrough_option('--k', type='int', help='Number of clusters')
    self.add_file_option('--centroids')


###########################################################################################
###########################################################################################
###########################################################################################

class Random_Initial_Centroids(MRJob):

  def steps(self):
    return [self.mr(
      mapper_init = self.randomize_centroids,
      mapper      = self.find_centroids
      )
    ]

  def randomize_centroids(self):
    import random
  
    k     = int(self.options.k)
    count = int(self.options.count)

    while True:
      centroid_positions = [random.randint(1, count) - 1 for i in range(k)]
      centroid_positions = set(centroid_positions)
      if len(centroid_positions) == k:
        self.centroid_positions = sorted(list(centroid_positions))
        self.counter = 0
        return

  def convert_string_to_vector(self, _line):
    return [float(i) for i in _line.split('\n')[0].split(',')[1:]]

  def find_centroids(self, _, _line):
    self.counter += 1

    if len(self.centroid_positions) > 0:      
      if self.counter == self.centroid_positions[0]:
        self.centroid_positions.pop(0)
        vector = self.convert_string_to_vector(_line)
        yield vector, None

  def __init__(self, args):
    MRJob.__init__(self, args)

  def configure_options(self):
    super(Random_Initial_Centroids, self).configure_options()
    self.add_passthrough_option('--k', type='int', help='Number of clusters')
    self.add_file_option('--count')

###########################################################################################
###########################################################################################
###########################################################################################

class Count_Vectors(MRJob):

  def steps(self):
    return [self.mr
      (
      mapper   = self.map_lines,
      combiner = self.combine_local,
      reducer  = self.aggregate_counts
      )
    ]

  def map_lines(self, _, _line):
    yield "lines", 1

  def combine_local(self, _key, _count):
    yield (_key, sum(_count))

  def aggregate_counts(self, _key, _count):
    yield (_key, sum(_count))

  def __init__(self, args):
    MRJob.__init__(self, args)

  def configure_options(self):
    super(Count_Vectors, self).configure_options()
    self.add_passthrough_option('--k', type='int', help='Number of clusters')

###########################################################################################
###########################################################################################
###########################################################################################

VECTOR_COUNT_FILE = "/tmp/emr.kmeans.vector_count"
CENTROID_FILE     = "/tmp/emr.kmeans.centroids"

###########################################################################################
###########################################################################################
###########################################################################################

def extract_centroids(_job, _runner):
  c = []
  for line in _runner.stream_output():
    key, value = _job.parse_output_line(line)
    # print key, value
    c.append(key)
  return c

def extract_count(_job, _runner):
  for line in _runner.stream_output():
    key, value = _job.parse_output_line(line)
    # print key, value
  return value

def write_to_disk(_data, _file):
  f = open(_file, "w")
  pickle.dump(_data, f)
  f.close()

def centroids_converged(_old_centroids, _centroids):
  old_centroids = set(map(tuple, _old_centroids))
  centroids     = set(map(tuple, _centroids))
  difference     = old_centroids ^ centroids # first_set.symmetric_difference(secnd_set)
  return False if len(difference) else True

###########################################################################################
###########################################################################################
###########################################################################################

if __name__ == '__main__':
  args = sys.argv[1:] # get the arguments passed from the command line

  job = Count_Vectors(args=args) #initialize this function with the configuratoina and pass the arguments
  with job.make_runner() as runner:
    runner.run() # now actually run the method
    count = extract_count(job, runner)

  job = Random_Initial_Centroids(args=args + ['--count=' + str(count)])
  with job.make_runner() as runner:
    runner.run() # now actually run the method
    old_centroids = extract_centroids(job, runner)
    write_to_disk(old_centroids, CENTROID_FILE)

  i = 0
  while True:
    i+=1
    # print "Iteration #%i" % i
    job = UpdateCentroids(args=args + ['--centroids='+CENTROID_FILE])
    with job.make_runner() as runner:
      runner.run()
      centroids = extract_centroids(job, runner)
      centroids.sort(key=lambda x: x[0])
      write_to_disk(centroids, CENTROID_FILE)

      if centroids_converged(old_centroids, centroids):
        break
      else:
        old_centroids = centroids


  print 'Convergence Iterations:', i
  print centroids