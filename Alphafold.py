#@title Input protein sequence(s), then hit `Runtime` -> `Run all`
from google.colab import files
import os
import re
import hashlib
import random
import glob
from sys import version_info
python_version = f"{version_info.major}.{version_info.minor}"

def add_hash(x,y):
  return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]

# 设置FASTA文件路径（修改为您的实际路径）
fasta_path = '/content/drive/MyDrive/xyl_seq_rmdup.fasta' #@param {type:"string"}

# 自动解析FASTA文件
def parse_fasta(path):
    with open(path) as f:
        entries = []
        entry = {}
        for line in f:
            if line.startswith(">"):
                if entry:
                    entries.append(entry)
                # 提取第一个字段作为jobname（例如：AAM39089.1）
                entry = {'jobname': line[1:].strip().split()[0], 'sequence': ''}
            else:
                entry['sequence'] += line.strip()
        if entry:
            entries.append(entry)
    return entries

# 循环处理每个序列
for entry in parse_fasta(fasta_path):
    original_jobname = entry['jobname'][:50]  # 原始jobname
    query_sequence = entry['sequence']
    
    # 新增序列长度检查
    if len(query_sequence) > 2500:
        skip_log = "/content/drive/MyDrive/skipped_sequences.txt"
        with open(skip_log, "a") as f:
            f.write(f"{original_jobname}\n")
        print(f"序列 {original_jobname} 长度超过2500，已跳过")
        continue  # 跳过超长序列

    # 构造通配符路径模式，匹配带数字后缀的版本
    clean_jobname = re.sub(r'[^\w]', '', original_jobname)  # 使用正则表达式移除特殊字符
    drive_result_pattern = f"/content/drive/MyDrive/{clean_jobname}*.zip"
    existing_files = glob.glob(drive_result_pattern)

    # 检查Google Drive中是否存在匹配文件
    if existing_files:
        print(f"检测到已存在结果文件 {existing_files[0]}，跳过处理")
        continue  # 跳过当前序列的处理

    jobname = original_jobname
    query_sequence = entry['sequence']

    import time  
    start_time = time.time()  # 新增开始时间记录
    !echo "Processing {jobname}..."

    num_relax = 1 #@param [0, 1, 5] {type:"raw"}
    #@markdown - specify how many of the top ranked structures to relax using amber
    template_mode = "none" #@param ["none", "pdb100","custom"]
    #@markdown - `none` = no template information is used. `pdb100` = detect templates in pdb100 (see [notes](#pdb100)). `custom` - upload and search own templates (PDB or mmCIF format, see [notes](#custom_templates))

    use_amber = num_relax > 0

    # remove whitespaces
    query_sequence = "".join(query_sequence.split())

    basejobname = "".join(jobname.split())
    basejobname = re.sub(r'\W+', '', basejobname)
    jobname = add_hash(basejobname, query_sequence)

    # check if directory with jobname exists
    def check(folder):
      if os.path.exists(folder):
        return False
      else:
        return True
    if not check(jobname):
      n = 0
      while not check(f"{jobname}_{n}"): n += 1
      jobname = f"{jobname}_{n}"

    # make directory to save results
    os.makedirs(jobname, exist_ok=True)

    # save queries
    queries_path = os.path.join(jobname, f"{jobname}.csv")
    with open(queries_path, "w") as text_file:
      text_file.write(f"id,sequence\n{jobname},{query_sequence}")

    if template_mode == "pdb100":
      use_templates = True
      custom_template_path = None
    elif template_mode == "custom":
      custom_template_path = os.path.join(jobname,f"template")
      os.makedirs(custom_template_path, exist_ok=True)
      uploaded = files.upload()
      use_templates = True
      for fn in uploaded.keys():
        os.rename(fn,os.path.join(custom_template_path,fn))
    else:
      custom_template_path = None
      use_templates = False

    print("jobname",jobname)
    print("sequence",query_sequence)
    print("length",len(query_sequence.replace(":","")))
    #@title Install dependencies
    import os
    
    USE_AMBER = use_amber
    USE_TEMPLATES = use_templates
    PYTHON_VERSION = python_version


        
    if not os.path.isfile("COLABFOLD_READY"):
      print("installing colabfold...")
      os.system("pip install -q --no-warn-conflicts 'colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold'")
      if os.environ.get('TPU_NAME', False) != False:
        os.system("pip uninstall -y jax jaxlib")
        os.system("pip install --no-warn-conflicts --upgrade dm-haiku==0.0.10 'jax[cuda12_pip]'==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
      os.system("ln -s /usr/local/lib/python3.*/dist-packages/colabfold colabfold")
      os.system("ln -s /usr/local/lib/python3.*/dist-packages/alphafold alphafold")
      os.system("touch COLABFOLD_READY")

    if USE_AMBER or USE_TEMPLATES:
      if not os.path.isfile("CONDA_READY"):
        print("installing conda...")
        os.system("wget -qnc https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh")
        os.system("bash Miniforge3-Linux-x86_64.sh -bfp /usr/local")
        os.system("mamba config --set auto_update_conda false")
        os.system("touch CONDA_READY")

    if USE_TEMPLATES and not os.path.isfile("HH_READY") and USE_AMBER and not os.path.isfile("AMBER_READY"):
      print("installing hhsuite and amber...")
      os.system(f"mamba install -y -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0 openmm=8.2.0 python='{PYTHON_VERSION}' pdbfixer")
      os.system("touch HH_READY")
      os.system("touch AMBER_READY")
    else:
      if USE_TEMPLATES and not os.path.isfile("HH_READY"):
        print("installing hhsuite...")
        os.system(f"mamba install -y -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0 python='{PYTHON_VERSION}'")
        os.system("touch HH_READY")
      if USE_AMBER and not os.path.isfile("AMBER_READY"):
        print("installing amber...")
        os.system(f"mamba install -y -c conda-forge openmm=8.2.0 python='{PYTHON_VERSION}' pdbfixer")
        os.system("touch AMBER_READY")
    #@markdown ### MSA options (custom MSA upload, single sequence, pairing mode)
    msa_mode = "mmseqs2_uniref_env" #@param ["mmseqs2_uniref_env", "mmseqs2_uniref","single_sequence","custom"]
    pair_mode = "unpaired_paired" #@param ["unpaired_paired","paired","unpaired"] {type:"string"}
    #@markdown - "unpaired_paired" = pair sequences from same species + unpaired MSA, "unpaired" = seperate MSA for each chain, "paired" - only use paired sequences.

    # decide which a3m to use
    if "mmseqs2" in msa_mode:
      a3m_file = os.path.join(jobname,f"{jobname}.a3m")

    elif msa_mode == "custom":
      a3m_file = os.path.join(jobname,f"{jobname}.custom.a3m")
      if not os.path.isfile(a3m_file):
        custom_msa_dict = files.upload()
        custom_msa = list(custom_msa_dict.keys())[0]
        header = 0
        import fileinput
        for line in fileinput.FileInput(custom_msa,inplace=1):
          if line.startswith(">"):
            header = header + 1
          if not line.rstrip():
            continue
          if line.startswith(">") == False and header == 1:
            query_sequence = line.rstrip()
          print(line, end='')

        os.rename(custom_msa, a3m_file)
        queries_path=a3m_file
        print(f"moving {custom_msa} to {a3m_file}")

    else:
      a3m_file = os.path.join(jobname,f"{jobname}.single_sequence.a3m")
      with open(a3m_file, "w") as text_file:
        text_file.write(">1\n%s" % query_sequence)
    #@markdown ### Advanced settings
    model_type = "auto" #@param ["auto", "alphafold2_ptm", "alphafold2_multimer_v1", "alphafold2_multimer_v2", "alphafold2_multimer_v3", "deepfold_v1", "alphafold2"]
    #@markdown - if `auto` selected, will use `alphafold2_ptm` for monomer prediction and `alphafold2_multimer_v3` for complex prediction.
    #@markdown Any of the mode_types can be used (regardless if input is monomer or complex).
    num_recycles = "3" #@param ["auto", "0", "1", "3", "6", "12", "24", "48"]
    #@markdown - if `auto` selected, will use `num_recycles=20` if `model_type=alphafold2_multimer_v3`, else `num_recycles=3` .
    recycle_early_stop_tolerance = "auto" #@param ["auto", "0.0", "0.5", "1.0"]
    #@markdown - if `auto` selected, will use `tol=0.5` if `model_type=alphafold2_multimer_v3` else `tol=0.0`.
    relax_max_iterations = 200 #@param [0, 200, 2000] {type:"raw"}
    #@markdown - max amber relax iterations, `0` = unlimited (AlphaFold2 default, can take very long)
    pairing_strategy = "greedy" #@param ["greedy", "complete"] {type:"string"}
    #@markdown - `greedy` = pair any taxonomically matching subsets, `complete` = all sequences have to match in one line.
    calc_extra_ptm = False #@param {type:"boolean"}
    #@markdown - return pairwise chain iptm/actifptm

    #@markdown #### Sample settings
    #@markdown -  enable dropouts and increase number of seeds to sample predictions from uncertainty of the model.
    #@markdown -  decrease `max_msa` to increase uncertainity
    max_msa = "auto" #@param ["auto", "512:1024", "256:512", "64:128", "32:64", "16:32"]
    num_seeds = 1 #@param [1,2,4,8,16] {type:"raw"}
    use_dropout = False #@param {type:"boolean"}

    num_recycles = None if num_recycles == "auto" else int(num_recycles)
    recycle_early_stop_tolerance = None if recycle_early_stop_tolerance == "auto" else float(recycle_early_stop_tolerance)
    if max_msa == "auto": max_msa = None

    #@markdown #### Save settings
    save_all = False #@param {type:"boolean"}
    save_recycles = False #@param {type:"boolean"}
    save_to_google_drive = False #@param {type:"boolean"}
    #@markdown -  if the save_to_google_drive option was selected, the result zip will be uploaded to your Google Drive
    dpi = 200 #@param {type:"integer"}
    #@markdown - set dpi for image resolution

    if save_to_google_drive:
      from pydrive2.drive import GoogleDrive
      from pydrive2.auth import GoogleAuth
      from google.colab import auth
      from oauth2client.client import GoogleCredentials
      auth.authenticate_user()
      gauth = GoogleAuth()
      gauth.credentials = GoogleCredentials.get_application_default()
      drive = GoogleDrive(gauth)
      print("You are logged into Google Drive and are good to go!")

    #@markdown Don't forget to hit `Runtime` -> `Run all` after updating the form.
    #@title Run Prediction
    display_images = True #@param {type:"boolean"}

    import sys
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from Bio import BiopythonDeprecationWarning
    warnings.simplefilter(action='ignore', category=BiopythonDeprecationWarning)
    from pathlib import Path
    from colabfold.download import download_alphafold_params, default_data_dir
    from colabfold.utils import setup_logging
    from colabfold.batch import get_queries, run, set_model_type
    from colabfold.plot import plot_msa_v2

    import os
    import numpy as np
    try:
      K80_chk = os.popen('nvidia-smi | grep "Tesla K80" | wc -l').read()
    except:
      K80_chk = "0"
      pass
    if "1" in K80_chk:
      print("WARNING: found GPU Tesla K80: limited to total length < 1000")
      if "TF_FORCE_UNIFIED_MEMORY" in os.environ:
        del os.environ["TF_FORCE_UNIFIED_MEMORY"]
      if "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ:
        del os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]

    from colabfold.colabfold import plot_protein
    from pathlib import Path
    import matplotlib.pyplot as plt

    # For some reason we need that to get pdbfixer to import
    if use_amber and f"/usr/local/lib/python{python_version}/site-packages/" not in sys.path:
        sys.path.insert(0, f"/usr/local/lib/python{python_version}/site-packages/")

    def input_features_callback(input_features):
      if display_images:
        plot_msa_v2(input_features)
        plt.show()
        plt.close()

    def prediction_callback(protein_obj, length,
                            prediction_result, input_features, mode):
      model_name, relaxed = mode
      if not relaxed:
        if display_images:
          fig = plot_protein(protein_obj, Ls=length, dpi=150)
          plt.show()
          plt.close()

    result_dir = jobname
    log_filename = os.path.join(jobname,"log.txt")
    setup_logging(Path(log_filename))

    queries, is_complex = get_queries(queries_path)
    model_type = set_model_type(is_complex, model_type)

    if "multimer" in model_type and max_msa is not None:
      use_cluster_profile = False
    else:
      use_cluster_profile = True

    download_alphafold_params(model_type, Path("."))
    results = run(
        queries=queries,
        result_dir=result_dir,
        use_templates=use_templates,
        custom_template_path=custom_template_path,
        num_relax=num_relax,
        msa_mode=msa_mode,
        model_type=model_type,
        num_models=5,
        num_recycles=num_recycles,
        relax_max_iterations=relax_max_iterations,
        recycle_early_stop_tolerance=recycle_early_stop_tolerance,
        num_seeds=num_seeds,
        use_dropout=use_dropout,
        model_order=[1,2,3,4,5],
        is_complex=is_complex,
        data_dir=Path("."),
        keep_existing_results=False,
        rank_by="auto",
        pair_mode=pair_mode,
        pairing_strategy=pairing_strategy,
        stop_at_score=float(100),
        prediction_callback=prediction_callback,
        dpi=dpi,
        zip_results=False,
        save_all=save_all,
        max_msa=max_msa,
        use_cluster_profile=use_cluster_profile,
        input_features_callback=input_features_callback,
        save_recycles=save_recycles,
        user_agent="colabfold/google-colab-main",
        calc_extra_ptm=calc_extra_ptm,
    )
    results_zip = f"{jobname}.result.zip"
    os.system(f"zip -r {results_zip} {jobname}")

    #@title Package and download results
    #@markdown If you are having issues downloading the result archive, try disabling your adblocker and run this cell again. If that fails click on the little folder icon to the left, navigate to file: `jobname.result.zip`, right-click and select \"Download\" (see [screenshot](https://pbs.twimg.com/media/E6wRW2lWUAEOuoe?format=jpg&name=small)).

    if msa_mode == "custom":
      print("Don't forget to cite your custom MSA generation method.")

    # 原始下载代码注释掉
    # files.download(f"{jobname}.result.zip")

    # 新增Google Drive复制功能
    drive_root = '/content/drive/MyDrive/'
    target_path = os.path.join(drive_root, f"{jobname}.result.zip")

    # 确保文件同步完成
    os.sync()
    if os.path.exists(f"{jobname}.result.zip"):
        # 使用系统命令复制文件
        os.system(f"cp {jobname}.result.zip {drive_root}")
        print(f"文件已复制到Google Drive: {target_path}")
    else:
        raise FileNotFoundError("压缩文件生成失败，无法复制到Google Drive")

    # 添加显式内存清理（针对连续运行）
    import gc
    gc.collect()
    if 'drive' in locals():
        del drive
        gc.collect()

    if save_to_google_drive == True and drive:
      uploaded = drive.CreateFile({'title': f"{jobname}.result.zip"})
      uploaded.SetContentFile(f"{jobname}.result.zip")
      uploaded.Upload()
      print(f"Uploaded {jobname}.result.zip to Google Drive with ID {uploaded.get('id')}")

    # 每完成一个预测后释放内存
    #del msa_model, model, opt
    #torch.cuda.empty_cache()

    #time.sleep(180)

    # 在预测完成后添加耗时统计
    elapsed_time = time.time() - start_time
    print(f"Prediction for {jobname} completed successfully in {elapsed_time:.2f} seconds")    # 假设预测完成后，输出预测完成信息
