from as_evaluation import calCoverageScore, evalArticleCoverage, evalArticleErrorScore
PR= {"a1":[1 , 2], "a2":[5], "a3": [6]}
    # ground truth regions for xth article
GTR= {"a1":[1 , 4 , 5], "a2":[3], "a3": [], "a4": []}
ACS, meanACS = evalArticleCoverage(PR, GTR)
print("ACS", ACS)
print("meanACS", meanACS)

AES, meanAES = evalArticleErrorScore(PR, GTR)
print("AES", AES)
print("meanAES", meanAES)
coordinates = [(2, 3), (1, 5), (4, 2), (2, 1), (3, 4), (1.5, 700),(2.9, -10000)]

sorted_coordinates = sorted(coordinates, key=lambda x: (x[0], x[1]))
print(sorted_coordinates)