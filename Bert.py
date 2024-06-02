from packet import *

class BertSST2Model(nn.Module): #succession
    def __init__(self, class_size, origin_output, adapter_hidden, pretrained_name="bert-base-uncased"):
        super(BertSST2Model, self).__init__()
        self.base_model = BertModel.from_pretrained(pretrained_name, return_dict=True) #当 return_dict=True 时，模型的输出将是一个包含所有这些部分的字典。
        # 模型名称识别：当您传递
        # 'bert-base-uncased'
        # 作为参数给from_pretrained函数时，函数会识别这是一个预定义的模型名称。
        # 配置加载：函数首先会加载与bert - base - uncased模型相对应的配置。这个配置定义了模型的架构，包括层数、隐藏层的大小、注意力头的数量等。
        # 权重下载：如果本地没有缓存该模型的权重，from_pretrained函数会从Hugging
        # Face模型库（一个远程服务器）下载预训练的权重文件。这些文件通常包括模型的权重和优化器状态（如果可用）。
        # 权重应用：下载完成后，函数会加载这些权重到BERT模型的相应层中。
        # 模型初始化：最后，函数会返回一个初始化的BertModel实例，该实例已经加载了预训练的权重，并且可以根据需要进行微调或用于推理。
        # 这个过程是自动化的，用户只需要提供模型名称，transformers库会处理剩下的事情。这是transformers库的一个主要优点，它简化了加载和使用预训练模型的过程。
        # for name, param in self.base_model.named_parameters():
        #     if 'pooler' not in name:  # 排除池化层，如果需要的话
        #         torch.nn.init.normal_(param, mean=0, std=0.01)

        self.adapter1 = nn.Sequential( # fine-tune through the adapter
            nn.Linear(origin_output, adapter_hidden),
            nn.ReLU(),
            nn.LayerNorm(adapter_hidden),  # Add layer normalization
            nn.Dropout(0.2),
            nn.Linear(adapter_hidden, origin_output),
            nn.LayerNorm(origin_output),  # Add layer normalization
        )

        self.adapter2 = nn.Sequential( ##why need two adapters
            nn.Linear(origin_output, adapter_hidden),
            nn.ReLU(),
            nn.LayerNorm(adapter_hidden),  # Add layer normalization
            nn.Dropout(0.2),
            nn.Linear(adapter_hidden, origin_output),
            nn.LayerNorm(origin_output),  # Add layer normalization
        )

        self.classifier = nn.Sequential(
            nn.Linear(origin_output * 2, origin_output),  #why 2 * origin_output
            nn.ReLU(),
            nn.Linear(origin_output, class_size),
        )

        self.init_weights()
        # self.freeze_params_encoder()

    def init_weights(self):  #可调参数随机初始化
        for module in [self.adapter1, self.adapter2, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear): #check the type
                    nn.init.xavier_uniform_(layer.weight) #random but scientific way
                    nn.init.zeros_(layer.bias)

    # nn.init.xavier_uniform_( nn.init.xavier_uniform_ 不是从正态分布中采样，而是从均匀分布中采样。Xavier初始化（也称为Glorot初始化）的目的是在训练开始时保持网络各层的激活值和梯度的方差大致相同，从而有助于网络的收敛。
    #     layer.weight): 这行代码使用了Xavier初始化（也称为Glorot初始化），它是一种用于初始化深度神经网络权重的科学方法。Xavier初始化的目的是保持网络在各层的输入和输出的方差大致相同，这有助于在训练开始时避免梯度消失或爆炸问题。该方法从均匀分布中抽取值，其范围取决于输入层的大小。
    # nn.init.xavier_uniform_: 这是PyTorch中的一个函数，用于将Xavier初始化应用于指定的权重矩阵（layer.weight）。
    # layer.weight: 这是要初始化的层的权重矩阵。
    # nn.init.zeros_(
    #     layer.bias): 这行代码将层的偏置项初始化为0。这意味着所有偏置都被设置为0，这通常适用于使用ReLU激活函数的情况，因为ReLU在输入为0或负数时输出为0，而当偏置为0时，它不会影响权重的初始线性表达。
    # nn.init.zeros_: 这是PyTorch中的一个函数，用于将0值初始化应用于指定的偏置向量（layer.bias）。
    # layer.bias: 这是要初始化的层的偏置向量。

    def freeze_params_encoder(self):
        for param in self.base_model.parameters(): #only change the classifier and adapter to save time
            param.requires_grad = False

    def forward(self, inputs):
        input_ids, input_tyi, input_attn_mask = inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]
        # 表示输入序列的编码表示；用于指示输入序列的不同部分，用于指示序列中的每个标记属于哪个部分；用于指示模型应该关注输入序列中的哪些部分
        outputs = self.base_model(input_ids=input_ids, attention_mask=input_attn_mask)
        bert_output = outputs.last_hidden_state[:, 0, :] #提取并存储BERT模型最后一层的隐藏状态。
        #这是BERT模型输出的一个属性，包含了模型最后一层的隐藏状态。这是一个三维张量，其形状通常是(batch_size, sequence_length, hidden_size)。

        adapter1_output = self.adapter1(bert_output) #add different but similar adpater to do downstream work
        adapter2_output = self.adapter2(bert_output)
        combined_output = torch.cat((adapter1_output, adapter2_output), dim=1)
        logits = self.classifier(combined_output) #why shall they cat together
        return logits