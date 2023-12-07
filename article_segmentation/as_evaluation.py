# -*- coding: utf-8 -*-
"""AS_Evaluation.ipynb

Author: Nancy Girdhar
Location: L3i, La Rochelle University, La Rochelle, France
Date: 15 January, 2023

## Article Error Rate
"""
import numpy as np

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list

def calCoverageScore(independentBlock,dependentBlock,predArticleBlockDict,articleBlockDict,blockDict2id):
  groundTruthDependent = list()
  groundTruthIndependent = list()
  for gk,gv in articleBlockDict.items():
    if gk!='' and len(gv) > 1:
      groundTruthDependent.append(gv)
    else:
      groundTruthIndependent.append(gv)
  
  groundTruthDependent    = sorted([element for innerList in groundTruthDependent for element in innerList])
  groundTruthIndependent  = sorted([element for innerList in groundTruthIndependent for element in innerList])
  
  if isinstance(groundTruthDependent[0], int) == False: 
    for i  in range(len(groundTruthDependent)):
      groundTruthDependent[i] = blockDict2id[groundTruthDependent[i]]

    for i  in range(len(groundTruthIndependent)):
      groundTruthIndependent[i] = blockDict2id[groundTruthIndependent[i]]
  
  #print("groundTruthDependent:",groundTruthDependent)
  #print("groundTruthIndependent:",groundTruthIndependent)  
  
  predictedDependent  = sorted(dependentBlock)
  predictedIndependent = list()
  for pk,pv in predArticleBlockDict.items():
    predictedIndependent.append(pv)
  
  predictedIndependent = [element for innerList in predictedIndependent for element in innerList]

  tempPredDependent = predictedIndependent 

  predictedIndependent = list(np.setdiff1d(predictedDependent, predictedIndependent))

  predictedDependent = tempPredDependent

  for b in independentBlock:
    predictedIndependent.append(b)
  
  predictedIndependent = sorted(predictedIndependent)

  #print("predictedDependent:",predictedDependent)
  #print("predictedIndependent:",predictedIndependent)

  commonDependent   = intersection(groundTruthDependent, predictedDependent)
  commonIndependent = intersection(groundTruthIndependent, predictedIndependent)

  #print("commonDependent:",commonDependent)
  #print("commonIndependent:",commonIndependent)

  ratioDependent = 0.0
  if len(groundTruthDependent) != 0:
    ratioDependent = len(commonDependent) / len(groundTruthDependent)
  
  rationIndependent = 0.0
  if len(groundTruthIndependent) != 0:
    rationIndependent = len(commonIndependent) / len(groundTruthIndependent)


  totalPredictedBlock = predictedDependent + predictedIndependent

  totalGroundTruthBlock = groundTruthDependent + groundTruthIndependent

  oversegmentation = 0.0
  if len(totalGroundTruthBlock) !=0:
    oversegmentation = len(totalPredictedBlock)/len(totalGroundTruthBlock)

  return oversegmentation


"""## Article Coverage Rate (ACR)"""

def evalArticleCoverage(predArticleBlockDict,groundTruthArticleBlockDict):
    """
    Input parameters:
    predArticleBlockDict= dictionary of predicted article with their respective regions(blocks)
    groundTruthArticleBlockDict = dictionary of ground truth article with their respective regions(blocks)

    This function calculates the coverage per article (ACR) ie. the correctly detected regions(blcoks) of that article compare to the ground truth and overall coverall of the articles in a page (mACR)
    """

    ACS = dict()
    for pa,pv in predArticleBlockDict.items():
        dif1 = np.setdiff1d(pv, groundTruthArticleBlockDict[pa])
        dif2 = np.setdiff1d(groundTruthArticleBlockDict[pa],pv)

        n_a = len(dif1) + len(dif2)
        d_a = len(Union(pv,groundTruthArticleBlockDict[pa]))
        
        ACS[pa] = 0.0
        if d_a != 0:
            ACS[pa] = 1 - (n_a / d_a)

    isum = 0.0
    for aesk,aesv  in ACS.items():
        isum = isum +  aesv
    
    meanACS = 0.0
    if len(ACS) != 0:
        meanACS = isum/len(ACS)

    return ACS,meanACS

def evalArticleErrorScore(predArticleBlockDict,groundTruthArticleBlockDict):
    """
    Input parameters:
    predArticleBlockDict= dictionary of predicted article with their respective regions(blocks)
    groundTruthArticleBlockDict = dictionary of ground truth article with their respective regions(blocks)

    This function calculates the coverage per article (ACR) ie. the correctly detected regions(blcoks) of that article compare to the ground truth and overall coverall of the articles in a page (mACR)
    """
    AES = dict()
    for pa,pv in predArticleBlockDict.items():
        dif1 = np.setdiff1d(pv, groundTruthArticleBlockDict[pa])
        dif2 = np.setdiff1d(groundTruthArticleBlockDict[pa],pv)

        n_a = len(dif1) + len(dif2)
        d_a = len(Union(pv,groundTruthArticleBlockDict[pa]))
        
        AES[pa] = 0.0
        if d_a != 0:
            AES[pa] = n_a / d_a

    isum = 0.0
    for aesk,aesv  in AES.items():
        isum = isum +  aesv
    
    meanAES = 0.0
    if len(AES) != 0:
        meanAES = isum/len(AES)

    return AES,meanAES

def evaluate(prediction, truth):
    if len(prediction) < len(truth):
        prediction = prediction + [[]]*(len(truth)-len(prediction))
    else:
        truth = truth + [[]]*(len(prediction)-len(truth))

    error_list = np.zeros((len(truth), len(prediction)))
    for truth_index, truth_value in enumerate(truth):
        for pre_index, pre_value in enumerate(prediction):
            jiaoji = set(truth_value).intersection(pre_value)
            bingji = set(truth_value + pre_value)
            error_list[truth_index][pre_index] = len(bingji-jiaoji) / len(bingji)

    error_value_list = np.min(error_list, axis=1)
    return error_value_list


