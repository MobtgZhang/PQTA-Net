set -e

echo "Download DuReader-checklist dataset"
wget --no-check-certificate https://dataset-bj.cdn.bcebos.com/lic2021/dureader_checklist.dataset.tar.gz
mkdir dureader_data
tar -zxvf dureader_checklist.dataset.tar.gz -C dureader_data
rm dureader_checklist.dataset.tar.gz
