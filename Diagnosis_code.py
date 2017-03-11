import math
from flask import Flask, render_template, request

class Tree:
   def __init__(self, parent=None):
      self.parent = parent
      self.children = []
      self.label = None
      self.classCounts = None
      self.splitFeatureValue = None
      self.splitFeature = None

    
def dataToDistribution(data):
   ''' Turn a dataset which has n possible classification labels into a
       probability distribution with n entries. '''
   allLabels = [label for (point, label) in data]
   #print("print all labeles")
   #print(allLabels) 
   numEntries = len(allLabels)
   #print("print all numentries")
   #print(numEntries) 
   possibleLabels = set(allLabels)
   #print("print all possibleLabels")
   #print(possibleLabels)

   dist = []
   for aLabel in possibleLabels:
      dist.append(float(allLabels.count(aLabel)) / numEntries)

   return dist

#print(dataToDistribution(data))


def entropy(dist):
   ''' Compute the Shannon entropy of the given probability distribution. '''
   return -sum([p * math.log(p, 2) for p in dist])


def splitData(data, featureIndex):
   ''' Iterate over the subsets of data corresponding to each value
       of the feature at the index featureIndex. '''

   # get possible values of the given feature
    
   
   attrValues = [point[featureIndex] for (point, label) in data]
   

   for aValue in set(attrValues):
      # compute the piece of the split corresponding to the chosen value
      dataSubset = [(point, label) for (point, label) in data
                    if point[featureIndex] == aValue]

      yield dataSubset


def gain(data, featureIndex):
   ''' Compute the expected gain from splitting the data along all possible
       values of feature. '''

   entropyGain = entropy(dataToDistribution(data))

   for dataSubset in splitData(data, featureIndex):
      entropyGain -= entropy(dataToDistribution(dataSubset))

   return entropyGain


def homogeneous(data):
   ''' Return True if the data have the same label, and False otherwise. '''
   return len(set([label for (point, label) in data])) <= 1


def majorityVote(data, node):
   ''' Label node with the majority of the class labels in the given data set. '''
   labels = [label for (pt, label) in data]
   choice = max(set(labels), key=labels.count)
   node.label = choice
   node.classCounts = dict([(label, labels.count(label)) for label in set(labels)])

   return node


def buildDecisionTree(data, root, remainingFeatures):
   ''' Build a decision tree from the given data, appending the children
       to the given root node (which may be the root of a subtree). '''

   if homogeneous(data):
      root.label = data[0][1]
      root.classCounts = {root.label: len(data)}
      return root

   if len(remainingFeatures) == 0:
      return majorityVote(data, root)

   # find the index of the best feature to split on
   bestFeature = max(remainingFeatures, key=lambda index: gain(data, index))

   if gain(data, bestFeature) == 0:
      return majorityVote(data, root)

   root.splitFeature = bestFeature

   # add child nodes and process recursively
   for dataSubset in splitData(data, bestFeature):
      aChild = Tree(parent=root)
      aChild.splitFeatureValue = dataSubset[0][0][bestFeature]
      root.children.append(aChild)

      buildDecisionTree(dataSubset, aChild, remainingFeatures - set([bestFeature]))

   return root


def decisionTree(data):
   return buildDecisionTree(data, Tree(), set(range(len(data[0][0]))))


def classify(tree, point):
   ''' Classify a data point by traversing the given decision tree. '''
   #print(tree.splitFeature)

   if tree.children == []:
      return tree.label
   else:
      matchingChildren = [child for child in tree.children
         if child.splitFeatureValue == point[tree.splitFeature]]
      #print(matchingChildren)  

      if len(matchingChildren) == 0:
         raise Exception("Classify is not able to handle noisy data. Use classify2 instead.")

      return classify(matchingChildren[0], point)

    


def dictionarySum(*dicts):
   ''' Return a key-wise sum of a list of dictionaries with numeric values. '''
   sumDict = {}

   for aDict in dicts:
      for key in aDict:
         if key in sumDict:
            sumDict[key] += aDict[key]
         else:
            sumDict[key] = aDict[key]

   return sumDict


def classifyNoisy(tree, point):
   ''' Classify a noisy data point by traversing the given decision tree.
       Return a dictionary of the appropriate class counts to account for
       multiple branching. '''

   if tree.children == []:
      return tree.classCounts
   elif point[tree.splitFeature] == '?':
      dicts = [classifyNoisy(child, point) for child in tree.children]
      return dictionarySum(*dicts)
   else:
      matchingChildren = [child for child in tree.children
         if child.splitFeatureValue == point[tree.splitFeature]]

      return classifyNoisy(matchingChildren[0], point)


def classify2(tree, point):
   ''' Classify data which is assumed to have the possibility of being noisy.
       Return the label corresponding to the maximum among all possible
       continuations of the tree traversal. That is, the maximum of the
       class counts at the leaves. Classify2 is equivalent to classify
       if the data are not noisy. If the data are noisy, classify will
       raise an error. '''
   counts = classifyNoisy(tree, point)

   if len(counts.keys()) == 1:
      return counts.keys()[0]
   else:
      return max(counts.keys(), key=lambda k: counts[k])


def testClassification(data, tree, classifier=classify2):
   ''' Test the classification accuracy of the decision tree on the given data set.
       Optinally choose the classifier to be classify or classify2. '''
   actualLabels = [label for point, label in data]
   predictedLabels = [classifier(tree, point) for point, label in data]

   correctLabels = [(1 if a == b else 0) for a,b in zip(actualLabels, predictedLabels)]
   return float(sum(correctLabels)) / len(actualLabels)


def testTreeSize(noisyData, cleanData):
   import random

   for i in range(1, len(cleanData)):
      tree = decisionTree(random.sample(cleanData, i))
      print (str(testClassification(noisyData, tree)) + ", ",)



   #print(data)
   #print(len(data[0][0]))

  # cleanData = [x for x in data if '?' not in x[0]]
   #print(cleanData)
   #print(noisyData)
  # noisyData = [x for x in data if '?' in x[0]]
   
    
   #print(data)

   #testTreeSize(noisyData, cleanData)
   

   
   
    
   
   #print (classify(tree, ['y' for _ in range(len(data[0][0]))]))
   #print (classify(tree, ['y','y','n','y','n','y','y','y','y','y','y','y','n'])) 
    
   #print (classify(tree, ['n' for _ in range(len(data[0][0]))]))

   
app = Flask(__name__)  

@app.route('/input', methods=['GET'])
def input():

  if request.method == 'GET':
    return render_template('user.html')
  
  
@app.route('/output', methods=['POST'])
def output():
  if __name__ == '__main__':
   with open('/Users/arjita/Desktop/Datasetfull.txt', 'r') as inputFile:
      lines = inputFile.readlines()
   print (lines)
   data = [line.strip().split(',') for line in lines]
   data = [(x[1:], x[0]) for x in data]

   print(data)
  
  tree = decisionTree(data)
  
  if request.method == 'POST':
    symptom = (request.form['symptom'])
    l1=["cough","sneeze","chest pain","chills","chronic fever","bad breadth","headache","dizziness","nausea","vomitting","dehydration","abdominal pain","loss of apetite"]
    l2=['n','n','n','n','n','n','n','n','n','n','n','n','n']

    for i in range(len(l1)):
      if l1[i]==symptom:
        l2[i]="y"
      else:
        l2[i]="n"


    disease=classify(tree,l2)     
            
    return render_template('output.html',SEND=disease)#,IMGSRC=colImgSrc)
      


# if __name__ == "__main__":
#   app.run()
