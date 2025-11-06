import torch, sys, os

import config
from modules.bert import BertForSequenceClassification
from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoConfig, AutoModelForMaskedLM, TorchAoConfig
import torchao.quantization as aoq
from torchao.dtypes import Int4CPULayout

device = torch.device("cpu")

# bert_clf = BertForSequenceClassification(vocab_size=30000).to(device)
# bert_clf.eval()

# print(bert_clf)


class MyBertConfig(PretrainedConfig):
    model_type = "mybert"

    def __init__(self,
                 vocab_size=30000,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 max_len=512,
                 dtype="f16",
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_len = max_len
        self.dtype  = dtype


class MyBertForMaskedLM(PreTrainedModel):
    config_class = MyBertConfig

    def __init__(self, config):
        super().__init__(config)
        # 这里你可以直接用你“手搓的”模型实现
        dtype_dict = {
            "f16": torch.float16,
            "f32": torch.float32,
        }
        
        self.bert_clf = BertForSequenceClassification(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_len=config.max_len,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            dtype=dtype_dict[config.dtype]
        )
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()  # 重要：HF会初始化权重用

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.bert_clf(input_ids, attention_mask=attention_mask)
        return logits
    
# 保存
config = MyBertConfig()
model = MyBertForMaskedLM(config)
model.save_pretrained("./mybert-hf")
config.save_pretrained("./mybert-hf")

# 注册
AutoConfig.register("mybert", MyBertConfig)
AutoModelForMaskedLM.register(MyBertConfig, MyBertForMaskedLM)

# # 加载
model = AutoModelForMaskedLM.from_pretrained("./mybert-hf")
model = model.to(device).eval()

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse, apply_classic_fusion
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp

torch.set_printoptions(precision=4,sci_mode=False)
x = torch.randn((1, 9, 768), dtype=torch.float16)

encoder = model.bert_clf.encoder.layer[0]
encoder = encoder.eval()

if len(sys.argv) == 2:
    if sys.argv[1] == "--w4only": quantizer =aoq.int4_weight_only(group_size=128, layout=Int4CPULayout())
    elif sys.argv[1] == "--w8a8": quantizer =aoq.int8_dynamic_activation_int8_weight()
    elif sys.argv[1] == "--w8only": quantizer =aoq.int8_weight_only()
    else: 
        raise Exception(f"{sys.argv[1]} unsupported. Supported quantization strategies are: --w4only, --w8only and --w8a8")
else:
    raise Exception("No quantization strategy specified")

aoq.quantize_(encoder, quantizer)

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

with torch.no_grad():
    graphs = dynamo_compiler.importer(encoder, x)
    
assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()
with open("./subgraph0.mlir", "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
    

# quantizer = aoq.int8_weight_only()
# aoq.quantize_(model, quantizer)
# print(model)

# import torch._dynamo as dynamo

# input_ids = torch.ones(1, 256).to(torch.long)
# attention_mask = torch.ones(1, 1, 256, 256).to(torch.float32)

# inputs = (input_ids, None, attention_mask)
# graph_module, guards = dynamo.export(bert_clf)(*inputs)

# print(graph_module.graph)
