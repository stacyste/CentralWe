import math

class BoltzmannValueIteration(object):
    def __init__(self, transitionTable, rewardTable, valueTable, convergenceTolerance, discountingFactor, beta):
        self.transitionTable = transitionTable
        self.rewardTable  = rewardTable
        self.valueTable = valueTable
        self.convergenceTolerance = convergenceTolerance
        self.gamma = discountingFactor
        self.beta = beta

    def __call__(self):
        
        delta = self.convergenceTolerance*100
        while(delta > self.convergenceTolerance):
            delta = 0
            for state, actionDict in self.transitionTable.items():
                valueOfStateAtTimeT = self.valueTable[state]
                qforAllActions = [self.getQValue(state, action) for action in actionDict.keys()]
                self.valueTable[state] = max(qforAllActions) 
                delta = max(delta, abs(valueOfStateAtTimeT-self.valueTable[state]))
        policyTable = {state:self.getBoltzmannPolicy(state) for state in self.transitionTable.keys()}

        return([self.valueTable, policyTable])
    
    def getBoltzmannPolicy(self, state, printStatments = False):
        exponents = [self.beta*self.getQValue(state, action) for action in self.transitionTable[state].keys()]
        actions = [action for action in self.transitionTable[state].keys()]

        # Scale to [0,700] if there are exponents larger than 700
        if len([exponent for exponent in exponents if exponent>700])>0:
            if printStatments:
                print("scaling exponents to [0,700]... On State:")
                print(state)
            exponents = [700*(exponent/max(exponents)) for exponent in exponents]

        statePolicy = {action: math.exp(exponent) for exponent, action in zip(exponents,actions)}
        normalizedPolicy = self.normalizeDictionaryValues(statePolicy)
        return(normalizedPolicy)

    def getQValue(self, state, action):
        nextStatesQ = [prob*(self.rewardTable[state][action][nextState] \
                             + self.gamma*self.valueTable[nextState]) \
                      for nextState, prob in self.transitionTable[state][action].items()]

        qValue = sum(nextStatesQ)
        return(qValue)
    
    def normalizeDictionaryValues(self, unnormalizedDictionary):
        totalSum = sum(unnormalizedDictionary.values())
        normalizedDictionary = {originalKey: val/totalSum for originalKey, val in unnormalizedDictionary.items()}
        return(normalizedDictionary)



def main():
    pass

if __name__ == '__main__':
    main()