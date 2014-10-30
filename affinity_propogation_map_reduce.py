from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import PickleProtocol, RawProtocol
import sys, random, numpy, pickle

##########################################################################################
##########################################################################################
##########################################################################################

class SimilarityDistances(MRJob):

  def mapper_init(self):
    self.vector_count = int(self.options.count)
    self.i = -1

  def mapper(self, _, _line):
    vector = [float(i) for i in _line.split('\t')[1].split(',')]
    self.i += 1
    for j in xrange(self.vector_count):
      yield (self.i,j), vector
      yield (j,self.i), vector

  def reducer(self, _key, _vectors):
    vectors        = numpy.array(list(_vectors))
    # if _key[0] != _key[1]:
    #   print _key, 0, vectors[0]
    #   print _key, 1, vectors[1]
    # difference     = vectors[0] - vectors[1]
    difference     = vectors[0] - vectors[1]
    sum_of_squares = numpy.sum(difference**2)

    yield (_key[0], _key[1]), -sum_of_squares

  def __init__(self, args):
    MRJob.__init__(self, args)

  def configure_options(self):
    super(SimilarityDistances, self).configure_options()
    self.add_file_option('--count')

###########################################################################################
###########################################################################################
###########################################################################################

class ArgmaxRow(MRJob):

  def mapper(self, _, _value):
    value    = _value.split('\t')
    address  = [int(i) for i in value[0][1:-1].split(", ")]

    row      = address[0]
    column   = address[1]
    yield row, (column, value[1])

  def reducer(self, _row, _values):
    convert_value = lambda value: ['', float(value)] if value[0] != 'c' else ['c', float(value[1:])]    
    maximum_arg   = lambda i, j: i if i[0] > j[0] else j
    maxi          = [float('-inf'),float('-inf')]

    for value in _values:
      column          = value[0]
      converted_value = convert_value(value[1])
      log_probability = converted_value[1]

      maxi = maximum_arg([log_probability, column], maxi)

    if converted_value[0] == '':
      # print _row, maxi[1]
      yield _row, maxi[1]
    else:
      # print _row, 'e' + str(maxi[1])
      yield _row, 'e' + str(maxi[1])


  def __init__(self, args):
    MRJob.__init__(self, args)

  def configure_options(self):
    super(ArgmaxRow, self).configure_options()

###########################################################################################
###########################################################################################
###########################################################################################

class SimilarityHistogram(MRJob):

  def mapper(self, _, _value):
    value = _value.split('\t')
    distance = float(value[1])
    yield distance, 1
    # # address  = [int(i) for i in value[0][1:-1].split(", ")]
    # if distance != 0:
    #   yield None, distance

  def combiner(self, _distance, _count):
    total = sum(list(_count))
    yield _distance, total

  def reducer(self, _distance, _count):
    total = sum(list(_count))
    # print _distance, total
    yield _distance, total

  def __init__(self, args):
    MRJob.__init__(self, args)

  def configure_options(self):
    super(SimilarityHistogram, self).configure_options()

###########################################################################################
###########################################################################################
###########################################################################################

class SimilarityMedian(MRJob):

  def mapper(self, _, _value):
    value = _value.split('\t')
    # distance = float(value[0])
    count    = float(value[1])
    yield _, count

  def combiner(self, _, _values):
    values = list(_values)
    values.sort()
    yield _, values

  def reducer(self, _, _values):
    values = list(_values)[0]
    values.sort()
    count  = len(values)
    if count % 2 == 0:
      upper_median = count / 2
      lower_median = count / 2 - 1
      median = (values[lower_median] + values[upper_median]) / 2
    else:
      median = (count - 1) / 2 + 1 
      median = values[median]
    yield _, median

  def __init__(self, args):
    MRJob.__init__(self, args)

  def configure_options(self):
    super(SimilarityMedian, self).configure_options()

###########################################################################################
###########################################################################################
###########################################################################################

# class SimilarityMedian(MRJob):

#   def mapper(self, _, _value):
#     value = _value.split('\t')
#     distance = float(value[1])
#     yield distance, 1
#     # # address  = [int(i) for i in value[0][1:-1].split(", ")]
#     # if distance != 0:
#     #   yield None, distance

#   def combiner(self, _distance, _count):
#     total = sum(list(_count))
#     yield _distance, total

#   def reducer(self, _distance, _count):
#     total = sum(list(_count))
#     yield _distance, total

#   def reducer(self, _, _distances):
#     mini, maxi = 0, float('-inf')

#     minimum = lambda i, j: i if i < j else j
#     maximum = lambda i, j: i if i > j else j

#     for distance in _distances:

#       value = float(distance)
#       mini  = minimum(value, mini)
#       maxi  = maximum(value, maxi)
#     yield mini, maxi

#   def __init__(self, args):
#     MRJob.__init__(self, args)

#   def configure_options(self):
#     super(SimilarityMedian, self).configure_options()

# ###########################################################################################
# ###########################################################################################
# ###########################################################################################

class Similarity(MRJob):

  def remove_degeneracies(self, _value):
    eps    = numpy.finfo(numpy.double).eps
    tiny   = numpy.finfo(numpy.double).tiny * 100
    random = self.random_state.randn(1)[0]
    return (eps * _value + tiny) * random

  def mapper_init(self):
    self.random_state = numpy.random.RandomState(0)
    self.median       = float(self.options.median)
    self.median       += self.remove_degeneracies(self.median)
    # self.min          = float(self.options.min)
    # self.max          = float(self.options.max)

  def mapper(self, _, _value):
    value    = _value.split('\t')
    address  = [int(i) for i in value[0][1:-1].split(", ")]
    distance = float(value[1])
    if distance == 0:
      distance = self.median
    else:
      distance += self.remove_degeneracies(distance)
    # print address, distance      
    yield address, distance

  def reducer(self, _address, _distance):
    for distance in _distance:
      # error = 
      # distance = _distance + error
      # print _address, 's' + str(distance)
      yield _address, 's' + str(distance)

  def __init__(self, args):
    MRJob.__init__(self, args)

  def configure_options(self):
    super(Similarity, self).configure_options()
    self.add_file_option('--median')
    self.add_file_option('--min')
    self.add_file_option('--max')

# S=S+(eps*S+realmin*100).*rand(N,N);
# randn('state',rns);

###########################################################################################
###########################################################################################
###########################################################################################

class ZeroMatrix(MRJob):

  def steps(self):
    return [self.mr(
      mapper      = self.map,
      reducer     = self.reduce,
      )
    ]

  def map(self, _, _count):
    count = int(_count)
    for i in xrange(count):
      for j in xrange(count):
        yield (i,j), 0
        # yield (i,j), 'a' + str(0)

  def reduce(self, _address, _values):
    for value in _values:
      yield _address, self.options.matrix + str(value)

  def __init__(self, args):
    MRJob.__init__(self, args)

  def configure_options(self):
    super(ZeroMatrix, self).configure_options()
    self.add_file_option('--matrix')

###########################################################################################
###########################################################################################
###########################################################################################

class Responsibility(MRJob):

  def mapper(self, _, _value):
    value    = _value.split('\t')
    address  = [int(i) for i in value[0][1:-1].split(", ")]
    distance = float(value[1])
    yield address[0], (address[1], distance)

  def reducer(self, _row, _values):
    minimum = lambda i, j: i if i < j else j
    maximum = lambda i, j: i if i > j else j

    maxi, columns = {}, {}
    # mini = {}
    for value in _values:
      column          = value[0]
      similarity      = value[1]
      columns[column] = similarity
      # row[column] = similarity

      max_values = sorted(maxi.keys())
      # min_values = sorted(mini.keys())
      if len(max_values) <= 1:
        maxi[similarity] = column
        # mini[similarity] = column
      else:
        if similarity > max_values[1] or similarity > max_values[0]:
          del maxi[max_values[0]]
          maxi[similarity] = column

        # if similarity < min_values[0] or similarity < min_values[1]:
        #   del mini[min_values[1]]          
        #   mini[similarity] = column

    max_values = sorted(maxi.keys())

    column = maxi[max_values[1]]
    # print (_row, column), 'r' + str(columns[column] - max_values[0])
    yield (_row, column), 'r' + str(columns[column] - max_values[0])
    del columns[column]

    for i in columns.keys():
      # print (_row, i), 'r' + str(columns[i] - max_values[1])
      yield (_row, i), 'r' + str(columns[i] - max_values[1])

  def __init__(self, args):
    MRJob.__init__(self, args)

  def configure_options(self):
    super(Responsibility, self).configure_options()

###########################################################################################
###########################################################################################
###########################################################################################

class Availability(MRJob):

  def mapper(self, _, _value):
    value          = _value.split('\t')
    address        = [int(i) for i in value[0][1:-1].split(", ")]
    responsibility = float(value[1])
    yield address[1], (address[0], responsibility)

  def reducer(self, _column, _values):
    availability, column_availability = {}, 0

    values = {}
    for value in _values:
      row            = value[0]
      responsibility = value[1]
      values[row]    = responsibility

      availability[row] = 0
      if responsibility > 0 or _column == row:
        column_availability += responsibility
        availability[row]    = -responsibility

    for i in range(len(values)):
      availability[i] += column_availability
      if i != _column and availability[i] > 0:
        availability[i] = 0
      yield (i, _column), availability[i]

  def __init__(self, args):
    MRJob.__init__(self, args)

  def configure_options(self):
    super(Availability, self).configure_options()

###########################################################################################
###########################################################################################
###########################################################################################

class AddMatrices(MRJob):

  def mapper(self, _, _value):
    value   = _value.split('\t')
    address = [int(i) for i in value[0][1:-1].split(", ")]
    value   = value[1]
    yield (address[0], address[1]), value

  def reducer_init(self):
    self.weight_a = float(self.options.weight_a)
    self.weight_b = float(self.options.weight_b)

  def reducer(self, _column, _values):
    total = 0
    for value in _values:

      matrix = value[0]
      if matrix == 'a' or matrix == 's' or matrix == 'r':
        value  = float(value[1:])
      else:
        value  = float(value)

      if matrix == 'a' or matrix == 's':
        total += value
      elif matrix == 'r':
        total += self.weight_a * value
      else:
        total += self.weight_b * value

    yield _column, total

  def __init__(self, args): 
    MRJob.__init__(self, args)

  def configure_options(self):
    super(AddMatrices, self).configure_options()
    self.add_file_option('--weight_a')
    self.add_file_option('--weight_b')

###########################################################################################
###########################################################################################
###########################################################################################

class CheckForConvergence(MRJob):
# takes old exemplars and new, if they aren't equal then show me the change

  def mapper(self, _, _value):
    value    = _value.split('\t')
    row      = value[0]
    exemplar = value[1]
    # print row, exemplar
    yield row, exemplar

  def reducer(self, _row, _values):
    values= []
    for value in _values:
      values.append(int(value[1:])) if value[0] == 'e' else values.append(int(value))

    if values[0] != values[1]:
      # print _row, (values[0], values[1])
      yield _row, (values[0], values[1])


  def __init__(self, args):
    MRJob.__init__(self, args)

  def configure_options(self):
    super(CheckForConvergence, self).configure_options()

##########################################################################################
##########################################################################################
##########################################################################################

class CleanData(MRJob):

  def mapper_init(self):
    self.i = 0

  def mapper(self, _, _line):
    if _line != '':
      vector = [float(i) for i in _line.split(',')[:]]
      yield self.i, vector
      self.i += 1

  def reducer(self, _key, _vectors):
    for vector in _vectors:
      vector = str(vector).strip('[]')
      yield _key, vector

  def __init__(self, args):
    MRJob.__init__(self, args)

  def configure_options(self):
    super(CleanData, self).configure_options()
    self.add_passthrough_option('--k', type='int', help='Number of clusters')

##########################################################################################
##########################################################################################
##########################################################################################

class CountVectors(MRJob):

  def mapper(self, _, _line):
    if _line != '':
      yield "lines", 1

  def combiner(self, _key, _count):
    yield (_key, sum(_count))

  def reducer(self, _key, _count):
    yield (_key, sum(_count))

  def __init__(self, args):
    MRJob.__init__(self, args)

  def configure_options(self):
    super(CountVectors, self).configure_options()
    self.add_passthrough_option('--k', type='int', help='Number of clusters')

###########################################################################################
###########################################################################################
###########################################################################################

class OrganizeClusters(MRJob):

  def mapper(self, _, _line):
    line = _line.split('\t')
    yield line[1], line[0]

  def combiner(self, _key, _vectors):
    yield _key, list(_vectors)

  def reducer(self, _key, _vectors):
    print _key, list(_vectors)
    yield _key, list(_vectors)

  def __init__(self, args):
    MRJob.__init__(self, args)

  def configure_options(self):
    super(OrganizeClusters, self).configure_options()
    self.add_passthrough_option('--k', type='int', help='Number of clusters')

###########################################################################################
###########################################################################################
###########################################################################################

class CopyExemplars(MRJob):

  def mapper(self, _, _value):
    value     = _value.split('\t')
    row       = value[0]
    exemplar  = value[1]
    yield row, exemplar

  def reducer(self, _row, _values):
    for value in _values:
      yield _row, 'e' + str(value)

  def __init__(self, args):
    MRJob.__init__(self, args)

  def configure_options(self):
    super(CopyExemplars, self).configure_options()

###########################################################################################
###########################################################################################
###########################################################################################

class RetrieveExemplars(MRJob):

  def mapper(self, _, _value):
    value = _value.split('\t')
    row   = value[0]
    if value[1][0] == 'e':
      if row == value[1][1:]:
        yield row, 'exemplar'
    else:
      vector = value[1]
      yield row, vector

  def reducer(self, _row, _values):
    values = []
    for value in _values:
      values.append(value)

    if len(values) == 2:
      for value in values:
        if value != 'exemplar':
          print str(_row) + ':', value
          yield _row, value

  def __init__(self, args):
    MRJob.__init__(self, args)

  def configure_options(self):
    super(RetrieveExemplars, self).configure_options()

###########################################################################################
###########################################################################################
###########################################################################################

def extract_data(_job, _runner):
  data = ""
  for line in _runner.stream_output():
    key, value = _job.parse_output_line(line)
    data += str(key) + "\t" + str(value) + "\n"
  return data

def extract_count(_job, _runner):
  value = 0
  for line in _runner.stream_output():
    key, value = _job.parse_output_line(line)
  return str(value) + "\n"

def write_to_disk(_data, _file):
  # import json

  f = open(_file, "w")
  f.write(_data)
  # json.dump(_data, f)
  # pickle.dump(_data, f)
  f.close()

def centroids_converged(_old_centroids, _centroids):
  old_centroids = set(map(tuple, _old_centroids))
  centroids     = set(map(tuple, _centroids))
  difference     = old_centroids ^ centroids # first_set.symmetric_difference(secnd_set)
  return False if len(difference) else True

def add_dampened_matrices(_matrices, _output_file, _dampen_percentage=0):
  if _dampen_percentage == -1:
    weights = ['--weight_a=1', '--weight_b=1']
  else:
    weights = [1 - _dampen_percentage, _dampen_percentage]
    weights = ['--weight_a=' + str(weights[0]), '--weight_b=' + str(weights[1])]

  job = AddMatrices(args=_matrices + weights)
  run_job(job, _output_file)

def run_job(_job, _output_file):
  with _job.make_runner() as runner:
    runner.run()
    data = extract_data(job, runner)
    write_to_disk(data, _output_file)


###########################################################################################
###########################################################################################
###########################################################################################

VECTORS                 = "/tmp/emr.affinity_propagation.vectors"
VECTOR_COUNT            = "/tmp/emr.affinity_propagation.vector_count"
SIMILARITY              = "/tmp/emr.affinity_propagation.similarity"
# HISTOGRAM               = "/tmp/emr.affinity_propagation.histogram"
# MIN_MAX                 = "/tmp/emr.affinity_propagation.min_max"
AVAILABILITY            = "/tmp/emr.affinity_propagation.availability"
NEW_AVAILABILITY        = "/tmp/emr.affinity_propagation.new_availability"
AVAILABILITY_SIMILARITY = "/tmp/emr.affinity_propagation.availability_similarity"
RESPONSIBILITY          = "/tmp/emr.affinity_propagation.responsibility"
NEW_RESPONSIBILITY      = "/tmp/emr.affinity_propagation.new_responsibility"
CRITERION               = "/tmp/emr.affinity_propagation.criterion"
NEW_CRITERION           = "/tmp/emr.affinity_propagation.new_criterion"
EXEMPLARS               = "/tmp/emr.affinity_propagation.exemplars"
NEW_EXEMPLARS           = "/tmp/emr.affinity_propagation.new_exemplars"
CHANGED_EXEMPLARS       = "/tmp/emr.affinity_propagation.changed_exemplars"
EXEMPLAR_VECTORS        = "/tmp/emr.affinity_propagation.exemplar_vectors"
CLUSTERS                = "/tmp/emr.affinity_propagation.clusters"

###########################################################################################
###########################################################################################
###########################################################################################

if __name__ == '__main__':

  data = sys.argv[1:] # get the arguments passed from the command line
  dampening_factor = 0.5
  verbose = True
  convergence_iterations = 16
  remaining_convergence_iterations = convergence_iterations

  # Execute parallel affinity propagation updates
  # e = np.zeros((n_samples, convergence_iter))


  if verbose == True: print 'Vectors - Prepare - Begun'
  job = CleanData(args=data)
  run_job(job, VECTORS)
  if verbose == True: print 'Vectors - Prepare - Finish'

  if verbose == True: print 'Vectors - Count - Begun'
  job = CountVectors(args=[VECTORS]) #initialize this function with the configuratoina and pass the arguments
  with job.make_runner() as runner:
    runner.run() # now actually run the method
    total_vector_count = extract_count(job, runner)
    write_to_disk(total_vector_count, VECTOR_COUNT)
  if verbose == True: print 'Vectors - Count - Finish'
  print 'Total Vectors:', total_vector_count

  if verbose == True: print 'Availability - Zero - Begun'
  job = ZeroMatrix(args=[VECTOR_COUNT] + ['--matrix=a'])
  run_job(job, AVAILABILITY)
  if verbose == True: print 'Availability - Zero - Finish'

  if verbose == True: print 'Responsibility - Zero - Begun'
  job = ZeroMatrix(args=[VECTOR_COUNT] + ['--matrix=r'])
  run_job(job, RESPONSIBILITY)
  if verbose == True: print 'Responsibility - Zero - Finish'

  if verbose == True: print 'Criterion - Zero - Begun'
  job = ZeroMatrix(args=[VECTOR_COUNT] + ['--matrix=c'])
  run_job(job, CRITERION)
  if verbose == True: print 'Criterion - Zero - Finish'

  if verbose == True: print 'Exemplars - Zero - Begun'
  job = ArgmaxRow(args=[CRITERION])
  run_job(job, EXEMPLARS)
  if verbose == True: print 'Exemplars - Zero - Finish'

  if verbose == True: print 'Similarities - Calculate Distances - Begun'
  job = SimilarityDistances(args=[VECTORS] + ['--count=' + str(total_vector_count)])
  run_job(job, SIMILARITY)
  if verbose == True: print 'Similarities - Calculate Distances - Finish'

  # print 'Similarity - Histogram - Begun'
  # job = SimilarityHistogram(args=[SIMILARITY])
  # run_job(job, HISTOGRAM)
  # print 'Similarity - Histogram - Finish'

  if verbose == True: print 'Similarity - Median - Begun'
  job = SimilarityMedian(args=[SIMILARITY])
  with job.make_runner() as runner:
    runner.run()
    median = extract_data(job, runner)
    median = median.split('\t')[1]
  if verbose == True: print 'Similarity - Median - Finish'

  # print 'Similarity - Find Min and Max - Begun'
  # job = SimilarityHistogram(args=[SIMILARITY])
  # run_job(job, HISTOGRAM)

  if verbose == True: print 'Similarity - Creation - Begun'
  job = Similarity(args=[SIMILARITY] + ['--median=' + str(median)])
  run_job(job, SIMILARITY)
  if verbose == True: print 'Similarity - Creation - Finish'

  for i in range(200):
    if verbose == True: print 'Similarity + Availability - Begun'
    add_dampened_matrices([AVAILABILITY, SIMILARITY], AVAILABILITY_SIMILARITY, -1)
    if verbose == True: print 'Similarity + Availability - Finished'
    
    if verbose == True: print 'Responsibility - Update - Begun'
    job = Responsibility(args=[AVAILABILITY_SIMILARITY])
    run_job(job, NEW_RESPONSIBILITY)
    if verbose == True: print 'Responsibility - Update - Finish'

    if verbose == True: print 'Responsibility - Dampen - Begun'
    add_dampened_matrices([NEW_RESPONSIBILITY, RESPONSIBILITY], RESPONSIBILITY, dampening_factor)
    if verbose == True: print 'Responsibility - Dampen - Finish'

    if verbose == True: print 'Availability - Update - Begun'
    job = Availability(args=[RESPONSIBILITY])
    run_job(job, NEW_AVAILABILITY)
    if verbose == True: print 'Availability - Update - Finish'

    if verbose == True: print 'Availability - Dampen - Begun'
    add_dampened_matrices([NEW_AVAILABILITY, AVAILABILITY], AVAILABILITY, dampening_factor)
    if verbose == True: print 'Availability - Dampen - Finish'

    if verbose == True: print 'Responsibility + Availability - Begun'
    add_dampened_matrices([RESPONSIBILITY, AVAILABILITY], NEW_CRITERION, -1)
    if verbose == True: print 'Responsibility + Availability - Finish'

    if verbose == True: print 'Counting Converged - Begun'
    job = ArgmaxRow(args=[NEW_CRITERION])
    run_job(job, NEW_EXEMPLARS)
    if verbose == True: print 'Counting Converged - Finish'

    job = CheckForConvergence(args=[NEW_EXEMPLARS,EXEMPLARS])
    run_job(job, CHANGED_EXEMPLARS)

    job = CountVectors(args=[CHANGED_EXEMPLARS])
    with job.make_runner() as runner:
      runner.run()
      changed_exemplar_count = extract_count(job, runner)

    if int(changed_exemplar_count) == 0:
      remaining_convergence_iterations -= 1

      job = RetrieveExemplars(args=[EXEMPLARS, VECTORS])
      run_job(job, EXEMPLAR_VECTORS)

      print 'Iterations Remaining Until Convergence:', remaining_convergence_iterations
    else:
      remaining_convergence_iterations = 3#convergence_iterations


    if remaining_convergence_iterations != 0:
      if verbose == True: print "Vectors Not Converged:", changed_exemplar_count
      job = CopyExemplars(args=[NEW_EXEMPLARS])
      run_job(job, EXEMPLARS)
    else:
      print "!!!!!!!!Converged!!!!!!!!"

      job = OrganizeClusters(args=[EXEMPLARS])
      run_job(job, CLUSTERS)

      job = CountVectors(args=[CLUSTERS])
      with job.make_runner() as runner:
        runner.run()
        exemplar_count = extract_count(job, runner)

      print "# of Exemplars:", exemplar_count

      job = RetrieveExemplars(args=[EXEMPLARS, VECTORS])
      run_job(job, EXEMPLAR_VECTORS)
      break