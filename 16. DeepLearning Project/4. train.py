import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer

from utils import load_mnist #utils 에 load_mnist가 있음
from utils import split_data
from utils import get_hidden_sizes

#이전에 만들었던 코드들을 조합해서 최종적으로 사용자가 사용하는 script 만듦

def define_argparser(): 
    p = argparse.ArgumentParser() #argparse함수는 사용자의 입력을 어떤 configuration으로 받아올수있게한다.

    p.add_argument('--model_fn', required=True) #모델의 파일이름. required = True는 무조건 있어야한다는 뜻
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1) #gpu에서 어디서 돌릴지.default는 0인데 cuda가 
                                                                                          #available하면 gpu쓸수있고 아니면(else) cpu가 default

    p.add_argument('--train_ratio', type=float, default=.8) #train과 valid를 나누는 ratio

    p.add_argument('--batch_size', type=int, default=256) #미니배치 사이즈. 256이 default 
    p.add_argument('--n_epochs', type=int, default=20) #수정할때는 이 코드를 고치는게 아니라 임시로 수정을 원하면 실행할때 고치는거임

    p.add_argument('--n_layers', type=int, default=5) #Layer 개수. 이 Layer개수를 입력하면 자동으로 hidden layer가 정해지도록 할거임
    p.add_argument('--use_dropout', action='store_true') #dropout쓸지말지. batchnorm이 default이라는뜻. dropout가 True가 되면 batchnorm이 꺼짐.
    p.add_argument('--dropout_p', type=float, default=.3) #dropout을 쓰면 애를 갖다씀

    p.add_argument('--verbose', type=int, default=1) #아까 어느정도 정보를 출력해줄건지에 대한 코드

    config = p.parse_args()

    return config

#이 return된 config가 밑의 main함수에 입력값으로 들어감

def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id) #device를 뭘쓸지. config.gpu_id를 쓰면 실제 입력값을 access할 수 있음. 사용자가 고치면 사용자가 쓴값이 들어있고 안고쳤으면 default값이 들어있음. gpu가 없으면 -1이므로 cpu가 되고 아니면 gpu가 됨

    #그다음 data loader가 uitls.py에 정의한 load_mnist 함수를 불러옴
    x, y = load_mnist(is_train=True, flatten=True) #그럼 x = (60000, 784), y = (60000,)벡터가 들어있을거임. 이걸 8대2로 random split을 해준다. flatten안하면 (60000,28,28)임. 근데 flatten해서 (60000,784)임 
    x, y = split_data(x.to(device), y.to(device), train_ratio=config.train_ratio) #split_data함수는 utils.py에 짜놓은 함수임

    print("Train:", x[0].shape, y[0].shape) 
    print("Valid:", x[1].shape, y[1].shape) 

    input_size = int(x[0].shape[-1])#애도 하드코딩 안함. 맨 마지막차원이 입력차원이라고 선언. flatten해서 784
    output_size = int(max(y[0])) + 1 #10기의 class있다는뜻

    model = ImageClassifier( #model.py에 있는 ImageClassifier 선언
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=get_hidden_sizes(input_size, #get_hidden_sizes는 utils.py에 들어있는 함수임.
                                      output_size,
                                      config.n_layers), #config.n_layer는 사용자가 layer를 몇개주냐에 따라 자동으로 number of layers보고 자동으로 n_layers(number layers)을 보고 등차수열로 hidden_layers을 쭉 만들어줌
        use_batch_norm=not config.use_dropout, #batchnorm의 반대 = dropout. droput 필요하면 dropout_p를 넣어준다
        dropout_p=config.dropout_p,
    ).to(device) #그렇게해서 imageclassifier만들고 to(device)으로 보내고 그다음 optimizer을 만든다
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()
    #이렇게 해서 model과 crit 다 만듬

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)

# 그 상황에서 우리는 trainer생성한다. 
    trainer = Trainer(model, optimizer, crit) #trainer객체를 받아서 밑에서 trine=실행한다

    trainer.train( #model, optimizer,crit받아서 trainer를 학습해라. 돌려라
        train_data=(x[0], y[0]), #train_data에는 x[0],y[0]를 튜플로넣어주고 
        valid_data=(x[1], y[1]), #valid에는 x[1],y[1]를 튜플로 넣어준다
        config=config 
    ) #config안에 epoch이 있으니깐 정해진 epoch만큼 돌고 끝난다. 그럼 train.model은 그때 lowest_validaton_loss 를 값을 가지고있을것이다.

    # Save best model weights.
    torch.save({ #저장할때는 torch.save함수를 쓴다. 애는 딕셔너리형태인 key와 value로 저장되는게 보통임.단지 예뻐서 보기좋아서 딕셔너리형태로 저장.
        'model': trainer.model.state_dict(), #trainer.model에는 lowest validation loss 당시에 어떤 weight 파라미터가 떠잇을건데 그걸 다시 state_dict()로 해서 가져온다. 
        'opt': optimizer.state_dict(), #adam 배웠을때 adam은 학습재개를 위해서는 저장돼있어야한다라고 배웠음, 우린 재개하진않을건데 혹시모르니깐 이때 optimizer을 저장한걸 가져온다
        'config': config, #그리고 config도 저장을 해야한다. 나중에 우리가 load_state할때 모델이 있어야하는데 모델생성할때 하이퍼 파라미터가 많이 들어감. 이런 하이퍼파라미터를 계산을 하기위해서는 학습할 당시 configuration을 알아야한다. 그래서 까먹지말고 config 써야한다. 
    }, config.model_fn) #어디에 저장할건지 파일명을 쓴다. 


if __name__ == '__main__':  
    config = define_argparser() # 처음으로 사용되는게 define_argparser()이라고 해서 configuration를 먼저 받아옴.
    main(config) #그다음 메인함수에다가 해당 config를 넣어줌으로써 학습이 실행됨.
