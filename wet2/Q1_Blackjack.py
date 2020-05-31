import numpy as np
import matplotlib.pyplot as plt


class BlackJackSolution:

    def __init__(self):
        self.YProbTot = {}
        for i in range(2, 12):
            self.YProbTot[i] = self.CalcProbYtot(i)
        self.PrintProbYtot()

    def ProbB(self, k):
        if k == 10:
            return 4/13
        else:
            return 1/13

    def CalcProbYtot(self, Y):
        ProbSumY = np.zeros(23) # init SumY probability to zero
        ProbSumY[Y] = 1  #initial SumY is first card Y
        for i in range(1, 9):  # maximum number of rounds - 8 (first card Y is 2 and in every round card given is 2)
            ProbSumYRound_i = np.zeros(23)
            ProbSumYRound_i[22] = ProbSumY[22]
            for Sum in range(1, 17):  # for each possible sum of cards smaller than 17
                for Bi in range(2, 12):  # for each possible card Bi(revealed on round i)
                    Sum_Bi = Sum + Bi
                    if Sum_Bi > 21:
                        ProbSumYRound_i[22] += ProbSumY[Sum] * self.ProbB(Bi)  # SumY=22 is equivalent to SumY>21
                    else:
                        ProbSumYRound_i[Sum_Bi] += ProbSumY[Sum] * self.ProbB(Bi)
            for Sum in range(17, 22):
                ProbSumYRound_i[Sum] += ProbSumY[Sum]
            ProbSumY = ProbSumYRound_i
        # The game ends when SumY is ether 17\18\19\20\21 or larger than 21(22) So Return these Probabilities
        ProbYtot = {17: ProbSumY[17], 18: ProbSumY[18], 19: ProbSumY[19], 20: ProbSumY[20], 21: ProbSumY[21], 22: ProbSumY[22]}
        return ProbYtot

    def PrintProbYtot(self):
        print("Probability of Y_tot|Y:")
        print("P(Y_tot|Y) |    17    |    18    |    19    |    20    |    21    |  >21(Loss for Dealer) ")
        print("------------------------------------------------------------------------------------------")
        for Y in range(2, 12):
            #pYtot = self.CalcProbYtot(Y)
            pYtot = self.YProbTot[Y]
            print("Y = {0}      |   {1}  |   {2}   |   {3}   |   {4}   |   {5}   |   {6}  ".format(Y, round(pYtot[17],3), round(pYtot[18],3), round(pYtot[19],3), round(pYtot[20],3), round(pYtot[21],3), round(pYtot[22],3)))


    def ValueForHit(self, X, Y, Values):
        ValHit = 0
        for B in range(2, 12):
            if(B>21-X): #Player's card sum exceeded 21
                ValHit -= self.ProbB(B)
            else:
                ValHit += Values[X+B-4, Y-2]*self.ProbB(B)
        return ValHit


    def ValueForStick(self, X, Y):
        ValStick = 0
        pYtot = self.YProbTot[Y]
        for Ytot in range (17, 21):
            if Ytot > X: #Dealer Reached Higher result than Player
                ValStick -= pYtot[Ytot]
            elif Ytot < X: #Dealer Reached Lower result than Player
                ValStick += pYtot[Ytot]
        ValStick += pYtot[22] #Dealer exceeeded 21
        return ValStick

    def ValueIteration(self, NumOfIterations):
        Value = np.zeros([18,10]) #init values
        for n in range(NumOfIterations):
            ValueNthIteration = Value
            for X in range(4,22):
                for Y in range(2,12):
                    ValueHit = self.ValueForHit(X, Y, ValueNthIteration)
                    ValueStick = self.ValueForStick(X, Y)
                    if(ValueHit > ValueStick):
                        Value[X-4, Y-2] = ValueHit
                    else:
                        Value[X-4, Y-2] = ValueStick
        return Value

    def GreedyPolicy(self, Value):
        Policy = np.zeros([18, 10], 'int')  # init actions - 0-stick 1-hit
        for Y in range(2, 12):
            for X in range(4, 22):
                ValueHit = self.ValueForHit(X, Y, Value)
                ValueStick = self.ValueForStick(X, Y)
                if (ValueHit > ValueStick):
                    Policy[X - 4, Y - 2] = 1 # 1-hit
                else:
                    Policy[X - 4, Y - 2] = 0 # 0-stick
        return Policy

    def PlotValue(self, Value):
        x = np.linspace(4, 21, 18)
        y = np.linspace(2, 11, 10)
        X, Y = np.meshgrid(x, y)
        Z = Value
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_ylabel("Dealer's Showing - Y")
        ax.set_xlabel("Player's Sum - X")
        ax.set_zlabel('Optimal Value')
        ax.plot_wireframe(X, Y, np.transpose(Z))
        plt.show()


    def PrintValue(self, Value):
        print("Value Reached By Value Iteration:")
        print("X  |4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|9|20|21")
        print("------------------------------------------------")
        for Y in range(2, 12):
            ValueStr=""
            for X in range(4, 22):
                ValueStr = ValueStr+'|'+str(Value[X-4,Y-2])
            print("Y={0}{1}".format(Y, ValueStr))

    def PlotPolicy(self, Policy):
        MinStickSum = 21*np.ones(10, 'int')  # init Min Sum of player's cards (X) for which the player chooses 'stick'
        for Y in range(2, 12):
            for X in range(4, 22):
                if (Policy[X-4, Y-2] == 0):  #0-stick
                    MinStickSum[Y-2] = X
                    break
        y = np.linspace(2, 11, 10)
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(y,MinStickSum)
        plt.ylim(4,21)
        plt.grid()
        plt.xticks(y)
        ax.set_xlabel("Dealer's Showing - Y")
        ax.set_ylabel("Min X For Policy To Choose 'Stick'")
        plt.show()


    def PrintPolicy(self,Policy):
        print("Greedy Policy derived from Optimal Value:")
        print("X  |4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|9|20|21")
        print("------------------------------------------------")
        for Y in range(2, 12):
            PolicyStr=""
            for X in range(4, 22):
                PolicyStr = PolicyStr + '|' + str(Policy[X-4, Y-2])
            print("Y={0}{1}".format(Y, PolicyStr))



if __name__ == "__main__":
    # training
    b = BlackJackSolution()
    # Find Optimal Value
    Value = b.ValueIteration(40)
    b.PlotValue(Value)
    # Derive Policy
    Policy = b.GreedyPolicy(Value)
    b.PlotPolicy(Policy)





