# BootstrapResampling.py

import random



class BootstrapResampling():
    """ This class  implements the non-parametric bootstrap resampling procedure discussed in class.
    """

    def getAverageBaselineScore(self, dataIn:list):
        """Given a list of dictionaries (dataIn) with key
            'baselineScore' (float), calculate the average baselineScore
            Example: [ {'question':"Question Text", 'answer':"Answer Text",
            'baselineScore':0.0, 'experimentalScore':1.0}, ... ]

            :param dataIn: List of dictionaries with key 'baselineScore'
            :return: Average 'baselineScore' across all elements in list.
        """

        list_a = []
        for i in range(0,len(dataIn)):
            for key, value in dataIn[i].items():
                if key == 'baselineScore':
                    list_a.append(value)

        return sum(list_a) / len(list_a)

    def getAverageExperimentalScore(self, dataIn:list):
        """Given a list of dictionaries (dataIn) with key
            'experimentalScore' (float), calculate the average baselineScore
            Example: [ {'question':"Question Text", 'answer':"Answer Text",
            'experimentalScore':0.0, 'experimentalScore':1.0}, ... ]

            :param dataIn: List of dictionaries with key 'experimentalScore'
            :return: Average 'experimentalScore' across all elements in list.
        """
        list_a = []
        for i in range(0, len(dataIn)):
            for key, value in dataIn[i].items():
                if key == 'experimentalScore':
                    list_a.append(value)
        return sum(list_a) / len(list_a)

    def createDifferenceScores(self, dataIn:list):
        """Given a list of dictionaries (dataIn) with keys 'baselineScore'
            and 'experimentalScore', calculate their difference scores
            (experimentalScore - baselineScore).
            Example: [ {'question':"Question Text", 'answer':"Answer Text",
            'experimentalScore':0.0, 'experimentalScore':1.0}, ... ]
            Example output: [1.0, ...]

            :param dataIn: List of dictionaries with float keys 'baselineScore', 'experimentalScore'
            :return: List of floats representing difference scores (experimental - baseline)
        """
        baseline = []
        experimental = []
        for i in range(0, len(dataIn)):
            for key, value in dataIn[i].items():
                if key == 'baselineScore':
                    baseline.append(value)
                if key == 'experimentalScore':
                    experimental.append(value)

        subtracted = list()
        for item1, item2 in zip(baseline,experimental):
            subtracted.append((item2 - item1))
        return subtracted

    def generateOneResample(self, differenceScores:list):
        """Randomly resamples the difference scores, to make a bootstrapped resample
            Example input: [0, 1, 0, 0, 1, 0, 1, 1, 0]
            Example output: [1, 0, 1, 0, 0, 1, 0, 1, 1]

            :param differenceScores: A list of difference scores (floats).
            :return: A list of randomly resampled difference scores (floats),
                of the same length as the input, populated using random
                sampling with replacement.
        """

        return random.choices(differenceScores, k=len(differenceScores))

    def calculatePValue(self, dataIn:list, numResamples=10000):
        """Calculate the p-value of a dataset using the bootstrap resampling procedure.
            Example: [ {'question':"Question Text", 'answer':"Answer Text",
            'baselineScore':0.0, 'experimentalScore':1.0}, ... ]
            Example output: 0.01

            :param dataIn: List of dictionaries with float keys 'baselineScore', 'experimentalScore' populated
            :param numResamples: The number of bootstrap resamples to use (typically 10,000 or higher)
            :return: A value representing the p-value using the bootstrap resampling procedure (float)
        """
        count = 0
        for i in range(0,numResamples):
            a = sum(self.generateOneResample(self.createDifferenceScores(dataIn)))
            if a <= 0:
                count = count +1
        return count / numResamples
