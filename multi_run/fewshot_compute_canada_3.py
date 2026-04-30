import pandas as pd
import json
import re
from tqdm import tqdm
from openai import OpenAI
#from sentence_transformers import SentenceTransformer
import os
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers import AutoModelForCausalLM, BitsAndBytesConfig,pipeline
import torch

datasets = ['abinit','lammps','libmesh','mdanalysis','zeng','zafar']
ROOT = os.path.join('/','home','ahsan','scratch')
data_path = os.path.join(ROOT,  'dataset')
embedding_dir =  os.path.join(ROOT,  'sentence_embedding')
response_dir = os.path.join(ROOT,  'responses')  

llm_model_list = ['meta-Llama-3.1-8B-Instruct',
                  'Qwen-Qwen2.5-72B',
                  'mistralai-Mixtral-8x22B-v0.1']

model_name_map_local = {
    'meta-Llama-3.1-8B-Instruct' : 'llama-3.3-70b',
    'Qwen-Qwen2.5-72B' : 'Qwen2.5-72B',
    'mistralai-Mixtral-8x22B-v0.1' : 'Mixtral-8x22B-v0.1'
}

example_commit_messages = {
  "lammps" : {'bug-fix': [("fix off-by-one memory allocation bug","de8d417aeccdcee893ae50ccb6f47fd41bcf802e"),
                          (r"""Bugfix: Kronecker term ignored in spline forces.
                              The code ignored the kronecker(ktype, 0) or kronecker(ltype, 0)
                              terms in the contributing terms to NconjtmpI and NconjtmpJ.
                              The issue was present both in ::bondorder and ::bondorderLJ and
                              led to energy conservation issues.
                              It has been fixed by checking for the atom type before entering
                              the offending calculations and adding clarifying comments.""","937cf0b996e5a31ebaeecbe412f03a4abaa508e0"),
                          ("Fixing more Kokkos per-atom and fdotr issues","1834a5e46c3a482df1740b3a5c37a07ffe277b67")
                         ],
              'not-bug-fix' : [
                  ("Fixing warning in pair_dpd_fdt_energy_kokkos", "85c8db5f86c7becfdb6c2d6831368abebabae0d4"),
                  (r"""kspace & dihedral can't do their own sync/modify
                  because the verlet_kokkos system has
                  a "clever" optimization which will
                  alter the datamasks before calling sync/modify,
                  so the datamask framework must be
                  strictly obeyed for GPU correctness.
                  (the optimization is to concurrently
                  compute forces on the host and GPU,
                  and add them up at the end of an iteration.
                  calling your own sync will overwrite
                  the partial GPU forces with the
                  partial host forces).""", "09fc8b0bd731270b2b08b720e6bb023eae86e707"),
                  (r"""Added ODE diagnostics to FixRxKokkos using Kokkos managed data.
                      - Added the diagnostics performance analysis routine to FixRxKokkos
                        using Kokkos views.
                      TODO:
                        - Switch to using Kokkos data for the per-iteration scratch data.
                          How to allocate only enouch for each work-unit and not all
                          iterations? Can the shared-memory scratch memory work for this,
                          even for large sizes?""", "4ac7a5d1f2e6132595c8999090e7b4159aa6971a")
              ]},
  "abinit" : {'bug-fix': [
                ("Correction of merge errors", "246774e1e726dd3430657f0ac6cdb82724604b25"),
                ("Try to workaround bug in ieig2rf (ebands_init)", "484acdf23351946ebd8552fb7ceeb7b9dcec0741"),
                (r"""Fixed mrgdv issue with large number of files

mrgdv used to open all POTx files simulatiously to read headers
and data in two seperate loops. However, some kernels only allow
1024 files to be open at the same time, freezing mrgdv if the
number of files is larger.

Merging the two loops, opening on file at a time and closing it
right after header and data are read and written fixes this issue.
The only drawback is that a corrupt file at the end of the loop
could waste a lot of time.""", "91fc156d12ee46e9f83b0e516eb2d28dd96b5f22")
              ],
              'not-bug-fix' : [
                  ("Some change to make HIST work with PIMD; new automatic test", "42b71a8b351edb148209bf2d970963f1f98bea65"),
                  ("Update setup_positron and its call", "61eb8e7cf8ae068b8e0460ce5f803ac1de656f2a"),
                  ("Fix a link, and update the var*.html files", "8cb11ba3d273d4c03f3bc6e99c21eee2a1c9843b")
              ]},
  "libmesh" : {'bug-fix': [
              ("corrections for --enable-ifem: InfFE and FE now use identical FEType to ease overall handling; added FEBase::build_InfFE; some typos", "d2e8cfa804ec243e6cb09bdccbedccb5a4e53aea"),
              (r"""partial fixes for InfFE; removed empty face_infinite.C; added some
explanatory comments for invalid calls to n_dofs etc; added an
example #5 -- better do not use it (yet), currently only for debugging
InfFE;
Had some problems with the build_cube() method, with higher-order
HEX elements: too many nodes initialized. Only gcc 295? don't know,
haven't tested...""", "4510a6e711cd5e211bfb1762d9459ad1e214b05a"),
            (r"""Patch from Vikram Garg implementing global error estimation and
an initial version of element-wise error indication based on refined
adjoint solutions""", "2d4e3aed5d6f32937ba1cb9d142b8b6b9ed83d34")
  ],
              'not-bug-fix' : [
                  (r"""SystemData now became GeneralSystem;
this works fine, but EquationSystems currently only handles a GeneralSystem;
added FrequencySystem (only an outline!);
documentation""", "6540bb9d34189ef9edafefd8e5d544580826752b"),
                   (r"""Add templated self friend classes so that Object<T1> can access
Object<T2>'s protected constructor



git-svn-id: file:///Users/petejw/Documents/libmesh_svn_bak@2621 434f946d-2f3d-0410-ba4c-cb9f52fb0dbf""", "0029bccd81a83a6f3e5bc832fe07864cc95c1192"),

                  ("extend blocksize option to matrices init()ed without the help of the DofMap","df2b0c5a44bc78430b4dcaa370dc1266bad0ec39")

              ]},
  "mdanalysis" : {'bug-fix': [
      (r"""Fixed some timeseries bugs - upgraded to work with numpy. Now individual timeseries objects can be treated like numpy arrays
This corresponds to r56 in /code/MDAnalysis.
git-svn-id: https://mdanalysis.googlecode.com/svn/trunk@47 dcee7792-c13e-0410-88f5-272e5f544b36""", "a49584c5772fb566edc9377a37d0116854e89022"),
        (r"""Issue 64 fixed: universe.trajectory.delta values are rounded when read from the dcd header file
(This used to be rounded to 4 decimals but people suggested that the full
precision would be better.""", "b946c380056b2738667ea6fef5920bb40fbf7d5d"),
("MDAnalysis now correctly writes float32 data for time and coordinates fields of ncdf files. Fixes #518", "e42c9c29708f5b4f64ad8f3ff14c8053fc60b19f")
  ],
              'not-bug-fix' : [
                  ("Removed unused CompositeSelection", "6d389eb6e6f0edc400a9ce6cafe5491465bb9c92"),
                  ("Moved the test for Issue#550 to its own class and improved as discussed in the PR thread.", "55bc758d93bb4f8fdf9c0048dd6d9d3c5bc590e8"),
                  ("DCDFile now allows overwriting, based on feedback in PR#1155.", "7bcec2666dd87fc5e86d6867bff3fa9ba420f4a4")
              ]},
  "zafar" : {
      'bug-fix': [
          ("[MXNET-144] Fix test_sparse_quadratic_function test (#10259)          * Add stream to a few FlatTo1D() calls          * fix build          * use enum in storage_initailized() in ndarray", "row-1454"),
          (r"""    Fixed ISSUE #170: "Provide ATOM 1.0 feeds instead of deprecated ATOM 0.3"          Submitted by:  Vladimir Sizikov
          git-svn-id: https://hudson.dev.java.net/svn/hudson/trunk/hudson/main@1245 71c3de6d-444a-0410-be80-ed276b4c234a""", "row-1646"),
          ("use list of musl pthreads files  instead of including them all by default  which mistakenly overrides JS impls", "row-467")
      ],
      'not-bug-fix' : [
          (r""" Create a TableExpansion when creating a TableNode.          This way the expandTables phase is no longer needed. We can also remove
          the expandOn function from TableNode and use the initial TableExpansion     everywhere.
          TableNode and TableExpansion are still separate Nodes so we can remove     TableExpansions after flattening all projections.""", "row-72"),
          ("Fix logger printout          Change-Id: I0d9845f52ee00c4fdf34bcea3b43dd83960bc9b2     Signed-off-by: Artem Barger <bartem@il.ibm.com>", "row-360"),
          (" Automated import from //branches/cupcake/...@142873 142873", "row-1134")
      ]
  },
  "zeng" : {
      'bug-fix': [
          ("chore(cubesql): Handle errors correctly for pg-wire","5185209e78b7c1cc75bb27746526b5c5599de50a"),
          ("fix: added valid string values to option enum properties, fixes #508", "b6328cf97a50e8cee736db0ac641f742cd09b38d"),
          (r"""refactor(search-bar): fix search icon so it doesn't cover the input field
          references #247""", "4ff55392ce68fb9ef2dea0c61024f11c2d22d6c8")
      ],
      'not-bug-fix' : [
          ("refactor!: policyhandler/handlenotification.go: rename armoResources to ksResources", "ed1862cf72b97f2e3fb60121513fdd2c8bd1451f"),
          ("fix(test): Moved playwright folder to nc-gui","957cbd2df6fa08450d664310ade6f7e820176e03"),
          ("fix(docz): add @emotion/core dependency", "52126df10f7110402429276fe4515e898a65eea9")
      ]
  }
}


def get_dataset(selected_data):
  print(f"Working on Dataset -> [{selected_data}]")
  df = pd.read_csv(os.path.join(data_path,f'{selected_data}_dataset.csv'))
  # Keep the format similar to all datasets. So, do some preprocessing.
  if ('zeng' == selected_data):
    # This is for the Zafar dataset. need special handling and renaming column like human fix
    df = df[['sha','commit_message', 'annotated_type']].dropna().reset_index(drop=True)
    df['binary_label'] = df['annotated_type'].apply(lambda x: 1 if x.strip().lower() == 'fix' else 0)
    df = df[['sha','commit_message', 'binary_label']].rename(columns={'sha':'commit_hash','commit_message': 'commit_msg', 'binary_label': 'human_fix'})
    #df.to_csv("zeng_clean.csv",index=False)
  elif ('zafar' == selected_data):
    df['commit_hash'] = ["row-" + str(x) for x in df.index]
    df.rename(columns={'text': 'commit_msg', 'label': 'human_fix'}, inplace=True)

  df = df[['commit_hash','commit_msg', 'human_fix']].dropna().reset_index(drop=True)
  df['human_fix'] = df['human_fix'].astype(int)

  return df

# Initialize the selected dataset
model_id = 0
selected_data = datasets[5]
df = get_dataset(selected_data)

import pickle

from numpy import select
# System prompt
system_prompt = "You are an expert developer and software engineering researcher. Given a commit message, your job is to classify the commit into either bug-fix or not bug-fix."

# user prompt for each commit message
definition_task = """Definition of a bug-fix commit:
A bug-fix commit refers to a change intended to correct an error, flaw, or unintended behavior in the software that causes it to produce incorrect or unexpected results, or to fail in some way. This includes fixing crashes, logical errors, performance issues, or addressing unintended behavior.

Task:
Analyze the commit message as input, return 1 if you classify it as a bug-fix commit. Otherwise, return 0 if it is not a bug-fix commit. This should be followed by a second number in the range 0-100 representing how confident you are in classifying the commit.

Follow the given examples:
"""
sel_data_examples = example_commit_messages[selected_data]

example_string = ""
id = 1
for x in sel_data_examples['bug-fix']:
    example_string = example_string + f"{id}. {x[0]}" + "\n"
    example_string = example_string + "Output: label : 1, confidence: 100" + "\n"
    id = id + 1
for x in sel_data_examples['not-bug-fix']:
    example_string = example_string + f"{id}. {x[0]}" + "\n"
    example_string = example_string + "Output: label: 0, confidence: 100" + "\n"
    id = id + 1

output_format = """
Output format (STRICT JSON):
{{
  "label": 1 if it is a bug-fix commit otherwise 0
  "confidence": integer (0-100),
  "rationale": "one-sentence summary of your rationale for classification",
}}

"""
user_prompt_template = """{definition} {example} {output}
Input Commit Message:
{commit_msg}
"""


def get_model(model_name):
    # Testing model loading and GPU memory usage
    
    # Define the local path where the model was downloaded
    local_model_path = os.path.join(ROOT, 'models', model_name)

    print("🔹 Starting tokenizer load...")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    #if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    print("✅ Tokenizer loaded.")

    print("🔹 Checking GPU memory before model load...")
    if torch.cuda.is_available():
        print('GPU is available. Checking memory usage...')
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"GPU Memory Cached:    {torch.cuda.memory_reserved()/1e9:.2f} GB")

    print("🔹 Starting model load...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        #load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        #torch_dtype=torch.float16
    )
    print("✅ Model loaded successfully on GPU.")

    return model, tokenizer

selected_model_name = llm_model_list[model_id]
print(f"Selected LLM Model: {selected_model_name}")
model, tokenizer = get_model(model_name_map_local[selected_model_name])   
model.eval()

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"temperature": 0.7, "do_sample": True, "max_new_tokens": 1000},
    device_map="auto"
)

# load pickle object
# Load sentence embedding dataset
output_file = os.path.join(embedding_dir, f"{selected_data}_embedding.pkl")
with open(output_file, 'rb') as f:
    loaded_data = pickle.load(f)

df_main = df.iloc[0:10]
error_instance = 0
json_response_file = os.path.join(response_dir, f"response_fewshots_{selected_data}_{selected_model_name}.jsonl")

for i, row in tqdm(df_main.iterrows(), total = len(df_main), desc="Building requests"):
    commit_hash = row['commit_hash']
    commit_msg = row['commit_msg']
    if commit_hash not in loaded_data:
        print("ERROR")
        continue
    user_prompt = user_prompt_template.format(
    definition = definition_task,
    example = example_string,
    output = output_format,
    commit_msg = commit_msg)

    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
    ]

    try:
        responses = generator(messages, pad_token_id=tokenizer.eos_token_id)
        text = responses[0]['generated_text'][-1]['content'].strip()
        
        print(f"prompt: {user_prompt} response {text}")
        msg_data = text.split("\n")

        response_data = {
            "custom_id" : f"commit-{i}",
            "commit_hash": commit_hash,
            "commit_msg": commit_msg,
            "model_response": text
        }

        # write the current response
        print(f"start writing file {json_response_file}")
        with open(json_response_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(response_data) + "\n")

    except Exception as e:
        error_instance = error_instance + 1
        print(f"An error occurred: {e}")


print(f"Error Instances {error_instance}")
print("Program finishes Successfully")
