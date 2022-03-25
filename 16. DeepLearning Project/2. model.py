import torch
import torch.nn as nn

#나중에  train.py 실행하면 하이퍼파라미터만 입력하면 자동으로 Layer 구성됨

#무조건 class 쓰면 def__init__() 설정부터 해야해!!

class Block(nn.Module):
    
    def __init__(self,
                 input_size,
                 output_size, 
                 use_batch_norm=True,
                 dropout_p=.4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        
        super().__init__()
        
        #########  regularizer부터 만들어!!! ###############
        ######### regularizer는 batch norm이니깐 use_batch_norm과 size가 들어가지!!!!!!!! ######################
        
        def get_regularizer(use_batch_norm, size): 
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)
        #Linear Layer 출력사이즈가 위 BatchNorm 사이즈 입력사이즈
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size), 
            nn.LeakyReLU(),
            get_regularizer(use_batch_norm, output_size), #dropout 또는 batchnorm 넣어준다. output_size가 nn.BatchNorm1d(output_size)임
        )
        
    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.block(x)
        # |y| = (batch_size, output_size)
        
        return y

    
class ImageClassifier(nn.Module): 

    def __init__(self,
                 input_size, #MNIST 가 Flatten되어 벡터로 들어오고 개 사이즈는 batch size 곱하기 input size이다
                 output_size, #output_size는 class별 각 log확률값을 뱉는다
                 hidden_sizes=[500, 400, 300, 200, 100], #사용자가 이 class을 선언할때 중간에 layer가 자동으로 생성됨. 즉 처음에 784 -> 500 -> 400..
                 use_batch_norm=True,
                 dropout_p=.3):
        
        #여기서 설정한거는 block 내부를 만드는데 사용할거임. block내부안에 5개 block 만들어
        
        super().__init__()

        assert len(hidden_sizes) > 0, "You need to specify hidden layers" #예외처리하는거임. hidden_size가 하나도 없으면 안된다
        # assert는 뒤의 조건이 True가 아니면 AssertError를 발생한다. 왜 assert가 필요한 것일까? 어떤 함수는 성능을 높이기 위해 반드시 정수만을 입력받아 처리하도록 만들 수 있다. 이런 함수를 만들기 위해서는 반드시 함수에 정수만 들어오는지 확인할 필요가 있다. 이를 위해 if문을 사용할 수도 있고 '예외 처리'를 사용할 수도 있지만 '가정 설정문'을 사용하는 방법도 있다.
       
        last_hidden_size = input_size
        blocks = [] 
        for hidden_size in hidden_sizes: # Block를 for문에 따라서 선언. Hidden layer을 자동으로 생성.
            blocks += [Block( #위에서 정의한 block함수에 넣기
                last_hidden_size, #첫번째 block size은 784에서 500감.두번째 for문돌때는 500에서 400
                hidden_size, #지금 hidden size는 500, 400, 300, 200, 100, 총 5개 잇음
                use_batch_norm,
                dropout_p
            )]
            last_hidden_size = hidden_size #그럼 500으로 업데이트됨. 400으로 나온값을 업데이트함
        
        self.layers = nn.Sequential(
            *blocks, # blocks와 함께 Linear layer와 Logsoftmax를 넣는다. 위에서의 결과값으로 blocks안에는 5개의 blocks가 나옴
            nn.Linear(last_hidden_size, output_size), #마지막으로 업데이트된 last_hidden size가 inpout으로 들어감. output은 각 class별 log확률뱉음
            nn.LogSoftmax(dim=-1),
        )
        
    def forward(self, x):
        # |x| = (batch_size, input_size)        
        y = self.layers(x)
        # |y| = (batch_size, output_size)
        
        return y

    #하드코딩없고 classification 한 general한 neural network를 만듦.