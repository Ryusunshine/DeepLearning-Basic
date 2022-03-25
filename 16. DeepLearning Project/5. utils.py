import torch


def load_mnist(is_train=True, flatten=True): #이미지 불러올때는 이 두 메소드가 필요함)
    from torchvision import datasets, transforms

    dataset = datasets.MNIST( #데이터 불러오기
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255. #0에서 1사이로 normalization
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y


def split_data(x, y, train_ratio=.8):
    train_cnt = int(x.size(0) * train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))
    x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    return x, y


def get_hidden_sizes(input_size, output_size, n_layers): #input_size는 784이고 output_size는 10인데 그 안에서 자동으로 hidden_layers가 등차수열로 계산해줄거임
    step_size = int((input_size - output_size) / n_layers) #step size = 등차값이 얼마냐

    hidden_sizes = []
    current_size = input_size
    for i in range(n_layers - 1): #이걸 쭉 돌려주면서 layers를 만든다
        hidden_sizes += [current_size - step_size]
        current_size = hidden_sizes[-1] #current_size에서 빼준 나머지값 hidden_size의 마지막차원에서 돌아가면서 빼줘야하니깐 hidden_size[-1]
    return hidden_sizes
