class Snakey():
    def __init__(self):
        #, epoch=10, batch_size=10, epsilon=1, gamma=.8
        self.thisfile = "snake-nn6.py"
        self.evaluateModel = False
        self.showVideo = True
        self.blocksize = 20
        self.epsilon = 0 #initial experimentation level
        self.epsilonIteration = 0 #when set to 0, gives random actions
        self.qgamma = 0.1 #learning rate of Q function
        self.gamma = 0.01 #Adam model learning rate
        self.epochs = 1
        self.outputEvery = 50 #only output every 50 cycles
        #self.failedScore = -450
        self.cycle = 0
        self.totalMoves = 0
        self.cycleBatch = 0 #counter
        #self.cycleBatchLimit = 50 #moves between training sessions
        self.experience = []
        self.snakeMemory = []
        self.trainedCount = 0
        self.filepath="snakels256.h5"


        self.inputfeatures = 12
        #self.minorEpoch = 1
        self.majorEpoch = 2
        self.batch_size = 999800
        self.loadWeights = False
        #self.weightsPath = "snake.h5"

        self.filepathtiny="snakeshort.h5"
        self.prediction = []
        self.highScore = 0
        self.highApples = 0
        self.iteration = 0
        #self.humanIterations = 0
        self.isRunning = True
        self.SCREEN_WIDTH = 500
        self.SCREEN_HEIGHT = 500
        if self.showVideo:
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption('Snake Ops')
            self.screen.set_alpha(None)

    def newSnake(self):
        #snake setup
        if self.showVideo:
            self.screen.fill(grey)
        self.snakeBody = np.array([[260, 240], [260,260]])
        self.x = self.snakeBody[0][0]
        self.y = self.snakeBody[0][1]
        self.foodExists = False
        self.food = [0,0]
        self.hasFed = False
        self.score = 0
        self.lastDirection = 0
        self.hasStarted = False
        self.collided = False
        self.cycle = 0
        self.apples = 0
        #self.snakeMemory = []   #short term memory for batch
        self.ensureFood()
        if self.showVideo:
            self.renderSnake()

    def renderSnake(self):
        if self.showVideo:
            for i in self.snakeBody:
                pygame.draw.rect(self.screen, white, (i[0], i[1], 19, 19))

    def ensureFood(self):
        if self.food[0] == 0 and self.food[1] == 0:
            self.food[0] = random.randint(0,24)*self.blocksize
            self.food[1] = random.randint(0,24)*self.blocksize
            while self.isPointOnSnake(self.food[0], self.food[1]):
                self.food[0] = random.randint(0,24)*self.blocksize
                self.food[1] = random.randint(0,24)*self.blocksize
            self.hasFed = False
        if self.showVideo:
            pygame.draw.rect(self.screen, red, (self.food[0], self.food[1], 19, 19))

    def showScore(self):
        if self.showVideo:
            fontlocation = str("/home/bander/apps/Python/snake-nn/veramono.ttf")
            fontlocation = fontlocation.replace(self.thisfile, '')
            font = pygame.font.Font(fontlocation, 18)
            string = str(int(self.score)) + " " + str(int(self.highApples)) + " " + str(int(self.iteration)) + " " + str(int(self.cycleBatch))# + " " + str(int(self.history.losses[-1]))
            text = font.render(string, True, white, grey)
            textRect = text.get_rect()
            textRect.center = (250, 20)
            self.screen.blit(text, textRect)


    def checkCollision(self):
        if self.isPointOnSnake(self.x, self.y):
            self.collided = True
        elif self.snakeBody[0][1] < 0 or self.snakeBody[0][1] > 480: #collide with y
            self.collided = True
        elif self.snakeBody[0][0] < 0 or self.snakeBody[0][0] > 480: #collided x
            self.collided = True


    def movePlayer(self, key):
        if key == 0:
            self.y -= 20
            self.lastDirection = 0
        elif key == 1:
            self.x += 20
            self.lastDirection = 1
        elif key == 2:
            self.y += 20
            self.lastDirection = 2
        elif key == 3:
            self.x -= 20
            self.lastDirection = 3
        else:
            if self.lastDirection == 0: self.y -= 20
            if self.lastDirection == 1: self.x += 20
            if self.lastDirection == 2: self.y += 20
            if self.lastDirection == 3: self.x -= 20


    def extendSnake(self):
        new_row = [self.x,self.y]
        self.snakeBody = np.insert(self.snakeBody, 0, new_row, axis=0)

    def feedOrDelete(self):
        if self.food[0] == self.snakeBody[0][0] and self.food[1] == self.snakeBody[0][1]:
            #if snake is eating now, delete last row
            self.food = [0,0]
            self.score += 100
            self.hasFed = True
        else:
            self.snakeBody = np.delete(self.snakeBody, -1, axis=0)

    def isPointOnSnake(self,x,y):
        result = False
        j=0
        for i in self.snakeBody:
            if i[0] == x and i[1] == y and j != 0:
                result = True
            j += 1
        return result

    def getSnakeState(self):
        #mup = self.lastDirection == 0  # move up
        #mright = self.lastDirection == 1  # move right
        #mdown = self.lastDirection == 2  # move down
        #mleft = self.lastDirection == 3  # move left
        fup = self.food[1] < self.y  # food up
        fright = self.food[0] > self.x  # food right
        fdown = self.food[1] > self.y  # food down
        fleft = self.food[0] < self.x  # food left
        xudanger = self.x <= 0 #or self.isPointOnSnake(self.x-self.blocksize,self.y) or self.lastDirection == 1
        xddanger = self.x >= 480 #or self.isPointOnSnake(self.x+self.blocksize,self.y) or self.lastDirection == 3
        yudanger = self.y <= 0 #or self.isPointOnSnake(self.y-self.blocksize,self.y) or self.lastDirection == 2
        yddanger = self.y >= 480 #or self.isPointOnSnake(self.y+self.blocksize,self.y) or self.lastDirection == 0
        xusnake = self.isPointOnSnake(self.x-self.blocksize,self.y)
        xdsnake = self.isPointOnSnake(self.x+self.blocksize,self.y)
        yusnake = self.isPointOnSnake(self.x,self.y-self.blocksize)
        ydsnake = self.isPointOnSnake(self.x,self.y+self.blocksize)
        #xproper = self.x == self.food[0]
        #yproper = self.y == self.food[1]
        #return  self.x, self.y, self.food[0], self.food[1],  mup, mright, mdown, mleft, angle,, xudanger, xddanger, yudanger, yddanger, angle, distance , distance, self.x, self.y, self.food[0], self.food[1],  mup, mright, mdown, mleft, angle , xproper, yproper
        state = [fup, fright, fdown, fleft, xudanger, xddanger, yudanger, yddanger, xusnake, xdsnake, yusnake, ydsnake]
        i=0
        for item in state:
            if item == True:state[i] = 1.0
            if item == False: state[i] = 0.0
            i += 1
        #print(state)
        return state


    def snakenn(self):
        #prep model saving
        #good model was LSTM128 d.1 Dense64 d.1 Dense64
        #best model LSTM256 d.1 D128 d.1 D128
        model = Sequential()
        #self.model.add(Dropout(0.2))
        model.add(LSTM(256, input_shape=(1,self.inputfeatures), return_sequences = True))
        #input_shape=(self.inputfeatures,)
        #model.add(Dense(64, activation='relu', input_shape=(1,self.inputfeatures)))
        #model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        #model.add(Dense(32, activation='relu'))
        #model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        #self.model.add(Dense(64, activation='relu'))
        #model.add(LSTM(128, return_sequences = True))
        model.add(Dropout(0.2))
        #model.add(LSTM(64, return_sequences = True))
        #self.model.add(Dropout(0.2))
        #self.model.add(LSTM(64, return_sequences = True))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(4, activation='linear'))
        #opt = Adam(self.gamma)
        #sgd = SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=False)
        model.compile(loss="mse", optimizer="adam")
        #model.compile(loss="categorical_crossentropy", optimizer=opt)
        #print(self.model.summary)
        return model
        #begin history for model
        #self.history = LossHistory()

    def remember(self):
        #agent.remember(state_old, final_move, reward, state_new, game.crash)
        self.experience = [self.state, self.action, self.reward, self.newstate, self.collided]
        self.snakeMemory.append(self.experience)


    def rewardSnake(self):
        #print(self.prediction)
        #self.reward = self.prediction[0][0][self.action]
        if self.collided == True:
            self.reward = -600
        elif self.hasFed == True:
            self.reward = 500
            self.apples += 1
        else:
            thisdistance = math.sqrt( ((self.x-self.food[0])**2)+((self.y-self.food[1])**2) )
            lastdistance = math.sqrt( ((self.snakeBody[1][0]-self.food[0])**2)+((self.snakeBody[1][1]-self.food[1])**2) )
            changedistance = lastdistance - thisdistance
            #distanceone = changedistance  #gets distance improved between -1 and 1
            #punish snake more for moving away from food, reward the close it is
            if changedistance < 0:
                changedistance *= 1.1
            self.reward = changedistance
        self.score += self.reward


    def trainShortTerm(self):
        target = self.reward
        #np.array(self.modeltiny.predict(np.reshape(self.state,(1, 1, self.inputfeatures))))
        if not self.collided:
            target = self.reward + self.qgamma * np.amax(np.array(self.modeltiny.predict(np.reshape(self.newstate,(1, 1, self.inputfeatures)))))
        target_f = np.array(self.modeltiny.predict(np.reshape(self.state,(1, 1, self.inputfeatures))))
        target_f[0][0][self.action] = target
        self.adjustedReward = target_f
        X = np.reshape(self.state, (1,1,self.inputfeatures)).astype(float)
        y = np.reshape(target_f, (1,1,4)).astype(float)
        self.modeltiny.fit(X, y, epochs=1, verbose=0)


    def trainLongTerm(self):
        if self.cycleBatch >= self.batch_size:
            memlength = len(self.snakeMemory)
            if memlength > self.batch_size:
                minibatch = random.sample(self.snakeMemory, self.batch_size)
            else:
                minibatch = self.snakeMemory
            memX = [];#states
            memY = [];#actionrewards
            for state, action, reward, newstate, done in minibatch:
                target = reward
                if not done:
                    target = reward + self.qgamma * np.amax(np.array(self.modeltiny.predict(np.reshape(self.newstate,(1, 1, self.inputfeatures)))))
                target_f = self.modeltiny.predict(np.reshape(state,(1, 1, self.inputfeatures)))
                target_f[0][0][action] = target
                memX.append(state)
                memY.append(target_f)
            memX = np.array(memX)
            memY = np.array(memY)
            print("train x y: ", memX[0], memY[0])
            X = np.reshape(memX, (memX.shape[0], 1,self.inputfeatures))
            y = np.reshape(memY, (memY.shape[0], 1,4))
            if self.evaluateModel: #get loss to continue training checkpoints based on best loss
                bestloss = self.modeltiny.evaluate(X, y, batch_size=len(memX))
                self.checkpoint.best = bestloss #set the checkpoints best loss value
                self.evaluateModel = False
            self.modeltiny.fit(X, y, epochs=1, batch_size=len(memX), verbose=1, callbacks=self.callbacks_list)#
            self.cycleBatch = 0
            #wipe all but the last 800 from long term memory
            if memlength > 2400:
                #print("memlength ", memlength)
                self.snakeMemory = self.snakeMemory[memlength-self.batch_size:]
                #print(len(self.snakeMemory))
            #self.modeltiny.save(self.filepath)
            #self.snakeMemory = []
            #self.modeltiny.set_weights(self.model.get_weights()) #copy this model to tiny model



    def playGame(self):
        self.newSnake()
        key = 0
        if self.loadWeights:
            self.model = load_model(self.filepath)
            self.modeltiny = load_model(self.filepath)
            #self.checkPointPath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
            self.evaluateModel = True #we need to reevaluate model to get loss
            print("loading models from ", self.filepath)
        else:
            self.model = self.snakenn()
            self.modeltiny = self.snakenn()
        self.checkpoint = ModelCheckpoint(self.filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        self.callbacks_list = [self.checkpoint]
        self.hasStarted = True
        while self.isRunning == True:

            while self.collided == False:
                # User Input - exit?
                #for event in pygame.event.get():
                #    if event.type == pygame.QUIT: self.isRunning = False
                #    if event.type == KEYDOWN:
                #        if event.key == K_ESCAPE: self.isRunning = False

                if self.showVideo:
                    self.screen.fill(grey)
                self.ensureFood()


                self.state = np.array(self.getSnakeState())
                self.prediction = np.array(self.modeltiny.predict(np.reshape(self.state,(1, 1, self.inputfeatures))))
                #self.prediction = np.array(self.model.predict(np.reshape(self.state,(1,1,self.inputfeatures)).flatten().tolist()))
                self.action = np.argmax(self.prediction)
                #print(self.action)
                #print(self.state, self.prediction)
                key = self.action


                #randomize key sometimes
                self.epsilon = 1 - (self.epsilonIteration / 75)
                if (random.random() < self.epsilon) and not self.loadWeights:
                    key = random.randint(0,3)
                    self.action = key


                #Move the snak based on key
                self.movePlayer(key)


                #snaks be snaks
                self.extendSnake()
                self.checkCollision()
                self.feedOrDelete()

                #get new state
                self.newstate = np.array(self.getSnakeState())

                #get reward for snake
                self.rewardSnake()

                #train short term memory on this state
                self.trainShortTerm()
                #agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)


                #remember this experience
                self.remember()


                #self.trainLongTerm()
                #print("snake: ", len(self.snakeMemory))

                #do every round
                self.renderSnake()
                self.cycle += 1
                self.cycleBatch += 1
                self.totalMoves += 1
                if self.score > self.highScore: self.highScore = self.score
                #push last stuff to display and flip
                self.showScore()
                if self.showVideo:
                    pygame.display.flip()

                if self.score < -400: self.collided = True


            #exited loop, rebegin
            if self.apples > self.highApples:
                self.highApples = self.apples
            print(str(self.iteration) + " HApple: " + str(int(self.highApples)) + " Apples: " + str(self.apples) + " Cycles: " + str(self.cycle) + " Totalmoves: " + str(self.totalMoves) + " Score: " + str(self.score)) #+ "Loss: " + self.history.losses[-1]
            #print("-----")
            self.iteration += 1
            self.epsilonIteration += 1
            self.trainLongTerm()
            self.newSnake()
            #print(self.history)
            #print("prediction, score: ", self.prediction, self.score)
        #save model before exit

        #print("model saved?")


# In[ ]:


if __name__ == '__main__':
    Snakey = Snakey()
    Snakey.playGame()
