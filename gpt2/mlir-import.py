import torch

import config
from modules.gpt2 import GPT2
from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoConfig, AutoModelForMaskedLM, TorchAoConfig
import torchao.quantization as aoq

device = torch.device("cpu")

# bert_clf = BertForSequenceClassification(vocab_size=30000).to(device)
# bert_clf.eval()

# print(bert_clf)

class MyGPT2Config(PretrainedConfig):
    model_type = "mygpt2"

    def __init__(self,
                 vocab_size=50257,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 max_len=1024,
                 dropout=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_len = max_len
        self.dropout = dropout


class MyGPT2ForMaskedLM(PreTrainedModel):
    config_class = MyGPT2Config

    def __init__(self, config):
        super().__init__(config)
        # 这里你可以直接用你“手搓的”模型实现
        self.gpt2 = GPT2(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_len=config.max_len,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            dropout=config.dropout
        )
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()  # 重要：HF会初始化权重用

    def forward(self, input_ids, attention_mask=None):
        logits = self.gpt2(input_ids, attention_mask=attention_mask)
        return logits


# 保存
config = MyGPT2Config()
model = MyGPT2ForMaskedLM(config)
model.save_pretrained("./mygpt2-hf")
config.save_pretrained("./mygpt2-hf")

# 注册
AutoConfig.register("mygpt2", MyGPT2Config)
AutoModelForMaskedLM.register(MyGPT2Config, MyGPT2ForMaskedLM)

# # 加载
model = AutoModelForMaskedLM.from_pretrained("./mygpt2-hf")
model = model.to(device).eval()
print(model.gpt2.transformer.h[0])

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse, apply_classic_fusion
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp

quantizer = aoq.int8_weight_only()
aoq.quantize_(model, quantizer)

x = torch.randn((1, 9, 768), dtype=torch.float16)

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

with torch.no_grad():
    graphs = dynamo_compiler.importer(model.gpt2.transformer.h[0], x)
    
assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()
with open("./subgraph0.mlir", "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
