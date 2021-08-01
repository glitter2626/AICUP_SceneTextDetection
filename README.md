0. 安裝所需套件:
* pip install numpy
* pip install scipy
* pip install pandas
* pip install opencv-python
* pip install scikit-image
* pip install json
* pip install matplotlib
1. 執行data_prepare.ipynb處理資料，並輸出COCO格式的json檔在TrainDataset、PublicTestDataset、PrivateTestDataset資料夾中。
2. 執行train_inference.ipynb訓練模型，訓練權重會儲存在mmocr-main/work-dirs/exp/，推理後的csv檔會儲存在當前目錄。
3. 模型訓練時的參數與讀取檔案的路徑皆寫在mmocr-main/configs/textdet/maskrcnn/目錄中的public_deform_mask_rcnn_101.py(for public test data推理)與private_deform_mask_rcnn_101.py(for private test data推理)，兩者除了推理檔案的路徑參數不同之外，其他參數皆相同
4. 工具庫運行架構可參考: https://mmdetection.readthedocs.io/en/latest/ 
5. 最佳模型參數檔案: https://drive.google.com/file/d/1BhXf53fwyO47uTt5FToFE8JwcG6OGTqI/view?usp=sharing
6. 檔案擺放如下:

![](https://i.imgur.com/WWiVdge.png)

  
