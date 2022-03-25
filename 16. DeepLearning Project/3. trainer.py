from copy import deepcopy
import numpy as np

import torch

class Trainer(): # Trainer안에 train함수있는거야

    def __init__(self, model, optimizer, crit): #학습하는데 필요한것들을 저장.
        self.model = model #지난시간에 만든 이미지 classifier을 넣음
        self.optimizer = optimizer #Adam쓸거임
        self.crit = crit #regression은 MSE LOSS 쓰닌깐 MSE Loss쓸거고 Classifier는 NLL 쓸거임

        super().__init__()

    def _batchify(self, x, y, batch_size, random_split=True): #SGD를 할거니깐 미니배치를 batchify해야함.
        # validation할때는 랜덤split안하니깐 random_split을 True/False로 지정하고서 validation 할때는 False로 입력함.
        if random_split: #랜덤 셔플링한이후에 랜덤 split을 함. random_split이 True인경우에만 셔플링하고 split. False는 그냥 split
            indices = torch.randperm(x.size(0), device=x.device) #x사이즈만큼 랜덤순열을 만들고 index_select. 
            # .device: 해당 tensor가 CPU, GPU 중 어디에 있는지 확인할 때 사용한다.
            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)
            #새롭게 지정된 x와 y는 랜덤하게 뒤섞인 상태임.데이터는 그대로고 순서만 뒤바뀐상태임. 같은기준으로 셔플링되어야함. 아니면 x와 y의 관계가 깨짐
        x = x.split(batch_size, dim=0) #그리고 batch_size로 균등하게 split함.
        y = y.split(batch_size, dim=0)

        return x, y
    

    def _train(self, x, y, config): #학습하면 self.model.train()부터 써주고 x,y 를 batchify해줘야함
        self.model.train() #꼭 써줘야함. model.trian()임. 시작할때 쓰면 편함. regularization을 train mode로 바꿔준다.

        x, y = self._batchify(x, y, config.batch_size) #우선 미니배치로 나눠준다. batchify함수에서 batchify된 x,y값을 반환.
        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)): #enumerate를 사용해서 x_i, y_i를 같이 출력. 그 x,y를 for문으로 i(몇번째 epoch)과 x_i, y_i(각 미니배치)를 가져옴
            y_hat_i = self.model(x_i) 
            #y_hat = fθ(x_i)
            #x_i의 사이즈는 (batch size, 차원(input_size)
            #y_hat_i의 사이즈는 (batch size, number class(=10 (output_size)(0~9까지 10개))
            # 원래 y_i는 (bs,)벡터임. Longtensor의 one-hot 인덱스만 들어잇음
            # loss = - 1/bs 합(y_i_t * y_hat_i)에서 y_i_transpose는 one-hot 인코딩임
            
            loss_i = self.crit(y_hat_i, y_i.squeeze()) #NLL을 이용해서 loss 구함

            # Initialize the gradients of the model.
            self.optimizer.zero_grad() #backpropagation하기전에 gradient를 0으로 초기화
            loss_i.backward()

            self.optimizer.step()

            if config.verbose >= 2: #사용자가 상황에따라서 많은정보를 보고싶거나 요약된 정보를 보고싶을때 매 iteration마다 정보를 보여줌
                print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

            # Don't forget to detach to prevent memory leak.
            total_loss += float(loss_i)#한 epoch의 평균 loss를 return

        return total_loss / len(x)


    def _validate(self, x, y, config):
        # Turn evaluation mode on.
        self.model.eval() # 까먹으면 안됨

        # Turn on the no_grad mode to make more efficintly.
        with torch.no_grad(): # 까먹으면 안됨
            x, y = self._batchify(x, y, config.batch_size, random_split=False) # _train 이나 _valid에 모두 있어야하는거야!
            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                if config.verbose >= 2: 
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

                total_loss += float(loss_i) 

            return total_loss / len(x)

    def train(self, train_data, valid_data, config): # 결과 총집합하는 곳, lowest_epoch, best_model 출력할거임
        #train + valid 합친 과정 (trainer-> validate 하는 총 과정을 뜻함). train, valid가 튜플로 올거임
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs): #매 epoch마다 train & validation 실행
            train_loss = self._train(train_data[0], train_data[1], config) #train_data[0] = x, train_data[1] = y
            valid_loss = self._validate(valid_data[0], valid_data[1], config) #애도 위에서 선언한 _validate함수에서 나온값을 매 epoch마다 불름

            # You must use deep copy to take a snapshot of current best weights.
            if valid_loss <= lowest_loss: #validation loss가 갱신하면 저장
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch_index + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))

        # Restore to best model.
        self.model.load_state_dict(best_model)

#앞서 코드를 살펴보기 전에 그림을 통해 전체 과정을 설명하였는데요. 그때 학습train과 검증validation 등을 아우르는 큰 루프loop가 있었고, 학습과 검증 내의 작은 루프가 있었습니다. train 함수 내의 for 반복문은 큰 루프를 구현한 것입니다. 따라서 내부에는 self._train 함수와 self._validate 함수를 호출하는 것을 볼 수 있습니다. 그리고 곧이어 검증 손실 값에 따라 현재까지의 모델을 따로 저장하는 과정도 구현된 것을 확인할 수 있습니다.        
        
        
        
        
        