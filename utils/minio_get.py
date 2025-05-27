
__version__ = "2.0.0"
import os
from minio import Minio
from minio.error import S3Error
from vllm_mxext.logger import init_logger
from progress.bar import Bar
import sys
import zipfile
import requests
from tqdm import tqdm
import requests
from modelscope.hub.snapshot_download import snapshot_download

logger = init_logger(__name__)

def get_object_with_progress(client, bucket_name, object_name, model_name):
    try:
        data = client.get_object(bucket_name, object_name)
        total_length = int(data.headers.get('content-length'))
        bar = Bar(object_name, max=total_length / 1024 / 1024, fill='*', check_tty=False,
                  suffix='%(percent).1f%% - %(eta_td)s')
        local_file_path = os.getenv("MIM_CACHE_PATH", "/opt/mim/.cache")
        model_path = os.path.join(local_file_path, model_name)
        os.makedirs(model_path, exist_ok=True)
        file_name = os.path.join(model_path, os.path.basename(object_name))
        with open(file_name, 'wb') as file_data:
            for d in data.stream(1024 * 1024):
                bar.next(1)
                file_data.write(d)
        bar.finish()
    except Exception as e:
        logger.error("downloading models from minio error: %s",str(e))
        
def unzip_and_delete(zip_path, extract_to):
    # 确保目标路径存在
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # 解压ZIP文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # 解压完成后删除ZIP文件
    os.remove(zip_path)
    logger.info(f"ZIP文件 '{zip_path}' 已解压到 '{extract_to}' 并删除。")
    
def download_from_minIO():
    #MinIO配置信息
    
    minio_url = "sw-s3-lg.metax-internal.com:9000"
    
    minio_access_key = os.getenv("MIM_ACCESS_KEY", "BuwNDEJz4tPJSOp3")
    minio_secret_key =  os.getenv("MIM_SECRET_KEY", "iVc31EUgJumJFrducSBNyaidhxJlzup0")
    bucket_name = "mim"
    local_file_path = os.getenv("MIM_CACHE_PATH", "/opt/mim/.cache")
    model_path = os.path.join(local_file_path, model_name)
    #判断/opt/mim/.cache目录下是否存在如果不存在则下载
    if( os.path.exists(model_path) and os.path.isdir(model_path)):
        logger.info(f"model:{model_name} exist at dir:{model_path}, no need to download.")
        return
    
    # 初始化MinIO客户端
    client = Minio(
        minio_url,
        access_key=minio_access_key,
        secret_key=minio_secret_key
    )
    
    objects = client.list_objects(bucket_name, prefix=model_name, recursive=True)
    for obj in objects:
        get_object_with_progress(client, bucket_name, obj.object_name, model_name)
        
def get_download_url(api_url, key):
    response = requests.get(api_url, headers={"Authorization": f"Bearer {key}"})
    if response.status_code != 200:
        logger.error(f"request for model download url call fun get_download_url fail, http errorcode:{response.status_code}")
        return None
    data = response.json()
    if data.get("code") != 0:
        logger.error(f"request for model download url call fun get_download_url fail, API errorcode:{data.get('message')}")
        return None
    return data.get("data", {}).get("url")

def download_model_by_url(url, filename):
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('Content-Length', 0))
    progress = tqdm(
        total=file_size,
        unit='iB',
        unit_scale=True,
        desc='Downloading'
    )

    if response.status_code == 200:
        with open(filename, 'wb') as file:
            progress.update(0)
            for chunk in response.iter_content(chunk_size=8192):
                size = file.write(chunk)
                progress.update(size)
        logger.info(f"model file saved:{filename}")
    else:
        logger.error(f"request download model failed. status code: {response.status_code}")

def download_model_from_modelscope(model_name: str, save_path: str):
    """
    从 ModelScope 下载指定模型，如果模型不存在或下载失败，返回错误信息。

    :param model_name: 要下载的模型名称（str）。
    :param save_path: 目标保存路径（str）。
    """
    try:
        # 检查目标路径是否存在，不存在则创建
        os.makedirs(save_path, exist_ok=True)
        
        # 开始下载模型
        logger.info(f"🔍 正在查询 ModelScope 以下载模型：{model_name}...")
        correct_path = os.path.join(save_path, model_name)
        if os.path.exists(correct_path):
            logger.info(f"✅ 模型 {model_name} 已经下载, 模型路径：{correct_path}")
            return 0
        # 使用 ModelScope 提供的 `snapshot_download` 方法
        model_dir = snapshot_download(model_name, cache_dir=save_path)
        
        logger.info(f"✅ 模型 {model_name} 下载完成，已保存到：{model_dir}")
        # 目标目录（使用原始模型名称）
        

        # 重命名目录（如果已经被修改）
        if os.path.exists(model_dir) and not os.path.exists(correct_path):
            os.rename(model_dir, correct_path)
        return 0
    except ValueError as e:
        logger.error(f"❌ 发生错误：模型 {model_name} 未找到！\n{e}")
        return 404
    except Exception as e:
        logger.error(f"❌ 下载过程中发生错误：\n{e}")
        return 401
        
def download_models(model_name:str):

    
    api_access_key = os.getenv("MIM_API_KEY", "ASODR4joTlQ50mir")
    local_file_path = os.getenv("MIM_CACHE_PATH", "/opt/mim/.cache")
    
    logger.info(f"starting to download model {model_name} to dir {local_file_path} from modelscope.  we trying to download it from metax office mim cloud platfrom")
    download_err = download_model_from_modelscope(model_name, local_file_path)
    if download_err != 0 :
        if download_err == 404:
            logger.warning(f"model {model_name} cannot find from modelscope.")
        if download_err == 401:
            logger.warning(f"some exception happened, so model {model_name} cannot download from modelscope. we trying to download it from metax office mim cloud platfrom")
        model_path = os.path.join(local_file_path, model_name)
        model_pkgname =  model_name + ".zip"
        model_packagename_dir = local_file_path + "/" + model_name + ".zip"
        #判断/opt/mim/.cache目录下是否存在如果不存在则下载
        if( os.path.exists(model_path) and os.path.isdir(model_path)):
            logger.info(f"model:{model_name} exist at dir:{model_path}, no need to download.")
            return
        api_url = f'https://mxs3.metax-tech.com/api/mx_download/v3/mim/{model_pkgname}'
        download_url = get_download_url(api_url, api_access_key)
        if download_url != None:
            download_model_by_url(download_url, model_packagename_dir)
            unzip_and_delete(model_packagename_dir, local_file_path)
        else:
            logger.error(f"Get model:{model_name} download url fail. API_KEY:{api_access_key} Http API:{api_url}")
    else:
        logger.info(f"download model {model_name} from modelscope successful.")
    
