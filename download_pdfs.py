import boto3
import os


ACCESS_KEY = '9dMgykbFMi1dXL67rGi9dQ'
SECRET_KEY = 'hxTcmcxCcLQASDjpz8Z1r3UDUyJ4XKvCfPsYGcE3mvvC'
ENDPOINT_URL = 'https://hb.ru-msk.vkcloud-storage.ru'
BUCKET_NAME = 'skifgold3-storage'


import boto3
import os

DOWNLOAD_DIR = "pdf_files"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

s3 = boto3.resource(
    's3',
    endpoint_url='https://hb.ru-msk.vkcloud-storage.ru',
    aws_access_key_id='9dMgykbFMi1dXL67rGi9dQ',
    aws_secret_access_key='hxTcmcxCcLQASDjpz8Z1r3UDUyJ4XKvCfPsYGcE3mvvC',
)

bucket = s3.Bucket('skifgold3-storage')

# Скачать все PDF
for obj in bucket.objects.all():
    if obj.key.endswith('.pdf'):
        local_path = os.path.join(DOWNLOAD_DIR, os.path.basename(obj.key))
        print(f"Скачиваю {obj.key} в {local_path} ...")
        bucket.download_file(obj.key, local_path)

