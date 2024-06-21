# Quality Assessment 

This repo contains experiments using various text-embedding and language models from the Azure OpenAI
service in Retrieval-Augmented Generation (RAG) scenarios. The goals of the project are to support
assessing the cost, accuracy, and latency of various models in support of determining which is a
best-fit for most RAG applications.

## Getting Started  

***Required:*** 
Your own data files and a 
subscription to Azure OpenAI Services
***

This repo is built using the Ubuntu Linux distro from Windows Subsystem for Linux v2. The first step
is getting a distro setup in Windows (we recommend using the Windows Terminal application, available
in the Microsoft Store). Once that's done, we need to configure the Git environment. The first step
is installing the [Git Credential Manager](https://github.com/git-ecosystem/git-credential-manager)
application. The easiest path is downloading the latest version from GitHub:

    https://github.com/git-ecosystem/git-credential-manager/releases/tag/v2.5.0

Next, you'll need to install the `pass` command:

```Bash
sudo apt install pass
```

Once that is downloaded, save it to the path `C:\Temp\gcm.tar.gz`. From there, in your WSL environment,
execute the following script:

```Bash
cp /mnt/c/Temp/gcm.tar.gz ./
sudo tar -xvf gcm.tar.gz -C /usr/local/bin
git-credential-manager
git clone ********************
git config --global credential.credentialStore gpg
git-credential-manager configure
pass init
gpg --gen-key
```

From there, you'll need to configure a `pass` environment:

```Bash
pass init <your-email>@email.com
```

## Set up 

This program allows you to upload your own data, modify specific testing parameters, and evaluate the performance of each experiment.

### Upload Your Data

Files are stored in the 'data' directory. You are able to add/remove files from the directory at any point. 

##### ***File Types:***

Files are pre-processed using the 'unstructured' library. It provides open-source components for ingesting and pre-processing images and text-documents.

Supported file types can be found here: https://docs.unstructured.io/open-source/core-functionality/partitioning 

To install, run the following command line: 

```bash
pip install "unstructured[all]"
```

### Prompts

User prompts can be written in the "prompt.txt" file. Program will automatically run through each question in the file. 

### Parameters

Experiment parameters can be modified in the 'experiment' dictionary at the top of the 'run_experiment.py' file. From there, you are able to add one or more values to the following parameters: (experiment name, llm models, embeddings models, k value, cosine similarity limit, chunk size, chunk overlap).

## Usage

### Run the Quality Assessment 

To execute the program based on your data and parameters: 

```bash
python3 run_experiment.py 
```

#### IMPORTANT

Program will not execute if "completions" folder already exists. Please move the completions from previous experiments before running the program. 

## Generated Output and Experiment Results 

### Completions 

Completions (prompts/responses) will be stored inside the "completions" directory, further organized into subdirectories based on the combination of the k value and cosine similarity limit of the experiment. There exists one query output file per LLM interaction. 

### Quality Assessment Results

Results will be stored ino the "table.csv" file in table format. 

## Naming Conventions 

### Database

The default VectorDB is Chroma. Created databases will be named as such: 

```
chroma_{embedding_model}_chunksize-chunkoverlap
```

### Completion Batches 

Completions will be named as such: 

```
llm-{gpt_model}.{embedding_model}.cs-{chunksize}.co-{chunkoverlap}.prompt-#
```
