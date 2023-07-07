[toc]

# 例子 Samples

## MNIST 手写数字识别

### 导入库

- 首先导入 tensorflow 库

        import tensorflow as tf

### 处理数据

- 加载 MNIST 数据集，并将样本数据从整数转换为浮点数

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

### 创建模型

- 在 `tf.keras.Sequential` 中堆叠网络层来构建模型

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])

- 对于每个样本，模型会输出一组分数，对应于可能的类别

        predictions = model(x_train[:1]).numpy()
        print(predictions)

- 使用激活函数 `tf.nn.softmax` 可以将这些分数转换为属于每个类的概率

        tf.nn.softmax(predictions).numpy()

- 使用 `losses.SparseCategoricalCrossentropy` 作为训练的损失函数，该损失等于正确的类的负对数概率，如果分类正确则损失为 0

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

- 开始训练之前，需要使用 `model.compile` 配置和编译模型，在这个将优化器 `optimizer` 设置为 `adam` 优化器，将损失函数 `loss` 设置为定义好的损失函数，并将 `metrics` 参数设置为 `accuracy` 来指定要为模型评估的指标

        model.compile(optimizer='adam',
                    loss=loss_fn,
                    metrics=['accuracy'])

### 训练模型

- 在 `model.fit` 中设置训练的循环次数进行训练，该过程中模型的参数不断更新，以最小化损失

        model.fit(x_train, y_train, epochs=5)

- 通过 `model.evaluate` 在验证集或者测试集上验证模型的性能

        model.evaluate(x_test,  y_test, verbose=2)

### 保存模型

一般情况下，训练模型需要花费很长的时间，所以需要将训练好的模型保存下来，而不是每次重新训练

        model.save('model.h5')

### 加载模型

        model = keras.models.load_model('model.h5')

### 进行预测

- 查看 MNIST 数据集的内容

        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5, 5, i+1)
            plt.imshow(x_test[i])
            plt.title(y_test[i])
            plt.axis("off")
        plt.show()

- 如果想让模型返回概率，需要封装经过训练的模型，并将激活函数 `softmax` 附加到该模型：

        probability_model = tf.keras.Sequential([
            model,
            tf.keras.layers.Softmax()
        ])
        print(probability_model(x_test[:5]))

## Fashion-MNIST 物品分类

### 处理数据

- Pytorch 有两个包负责数据处理，一个是 `torch.utils.data.Dataset` 负责存储数据和对应标签，一个是 `torch.utils.data.DataLoader` 负责遍历读取 `Dataset`

        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets
        from torchvision.transforms import ToTensor
        import matplotlib.pyplot as plt

- Pytorch 有三个包分别存储不同领域的数据集，分别是存储文本数据集的 `torchtext`，存储图像数据集的 `torchvision` 和存储声音数据集的 `torchaudio`
- 每一个数据集都有两个函数 `transform` 和 `target_transform`，前者用于对样本数据进行处理，后者用于对标签数据进行处理

        # Download training data from open datasets.
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

- 将 `Dataset` 作为参数传入 `DataLoader`，将创建一个对应数据集的迭代器，实现批量处理、采样、随机打乱和并行读取等功能

        batch_size = 64

        # Create data loaders.
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        for X, y in test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

### 创建模型

- 在 Pytorch 中构建模型，需要创建一个 `nn.Module` 类的子类，之后在 `__init__` 函数中设计网络层，在 `forward` 函数中设计数据传输流程

        # Get cpu, gpu or mps device for training.
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")

        # Define model
        class NeuralNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.linear_relu_stack = nn.Sequential(
                    nn.Linear(28*28, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 10)
                )

            def forward(self, x):
                x = self.flatten(x)
                logits = self.linear_relu_stack(x)
                return logits

        model = NeuralNetwork().to(device)
        print(model)

### 训练模型

- 训练模型的过程，本质上是优化更新网络参数，使得输出的结果更加接近目标，所以需要定义损失函数和优化器。损失函数的作用是度量结果和目标的差距，优化器的作用是按照一定的策略更新网络参数

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

- 在训练过程的一次循环中，模型先对训练数据做出预测，再回传预测误差，然后更新网络传输

        def train(dataloader, model, loss_fn, optimizer):
            size = len(dataloader.dataset)
            model.train()
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)

                # Compute prediction error
                pred = model(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if batch % 100 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

- 同时为了检验模型是否成功学习，需要在测试集上检验其预测结果

        def test(dataloader, model, loss_fn):
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            model.eval()
            test_loss, correct = 0, 0
            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

- 训练过程将会被重复多次执行，重复过程称为 epoch。在每个 epoch 中，我们可以输出模型的准确率和损失值，准确率上升和损失值下降表明这是成功的训练

        epochs = 5
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            test(test_dataloader, model, loss_fn)
        print("Done!")

### 保存模型

- 保存模型的一种方式，是将其内部状态的词典整个保存下来

        torch.save(model.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")

### 加载模型

- 加载模型的步骤包括重新创建模型结构，和读取状态词典中的参数

        model = NeuralNetwork().to(device)
        model.load_state_dict(torch.load("model.pth"))

### 进行预测

- 查看 Fashion-MNIST 数据集的内容

        classes = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]

        figure = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3
        for i in range(1, cols*rows+1):
            sample_idx = torch.randint(len(test_data), size=(1,)).item()
            img, label = test_data[sample_idx]
            figure.add_subplot(rows, cols, i)
            plt.title(classes[label])
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")
        plt.show()

- 模型加载之后就可以进行预测

        model.eval()
        x, y = test_data[0][0], test_data[0][1]
        with torch.no_grad():
            x = x.to(device)
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
