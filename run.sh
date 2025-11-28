### train
CUDA_VISIBLE_DEVICES="2" python tools/lazyconfig_train_net.py --config-file projects/ViTDet/configs/VOC2007coco/mask_rcnn_vitdet_b_100ep_dbias_6class.py \
    train.output_dir="output/voc2007_6class_from0"
CUDA_VISIBLE_DEVICES="2" python3.7 tools/lazyconfig_train_net_incre.py --config-file projects/ViTDet/configs/VOC2007coco/mask_rcnn_vitdet_b_100ep_dbias_clipart_6class.py \
     train.init_checkpoint=output/voc2007_6class_from0/model_final.pth train.output_dir="output/clipart_6class_frompa_domainbias"
CUDA_VISIBLE_DEVICES="2" python3.7 tools/lazyconfig_train_net_incre.py --config-file projects/ViTDet/configs/VOC2007coco/mask_rcnn_vitdet_b_100ep_dbias_watercolor.py \
     train.init_checkpoint=output/clipart_6class_frompa_domainbias/model_final.pth train.output_dir="output/watercolor_fromc_domainbias"
CUDA_VISIBLE_DEVICES="2" python3.7 tools/lazyconfig_train_net_incre.py --config-file projects/ViTDet/configs/VOC2007coco/mask_rcnn_vitdet_b_100ep_dbias_comic.py \
     train.init_checkpoint=output/watercolor_fromc_domainbias/model_final.pth train.output_dir="output/comic_fromw_domainbias"

### model combine ### lazyconfig_train_net_cbmodel.py need to be modified
CUDA_VISIBLE_DEVICES="4" python tools/lazyconfig_train_net_cbmodel.py --config-file projects/ViTDet/configs/VOC2007coco/mask_rcnn_vitdet_b_100ep_dbias_6class.py --eval-only
CUDA_VISIBLE_DEVICES="4" python tools/lazyconfig_train_net_cbmodel.py --config-file projects/ViTDet/configs/VOC2007coco/mask_rcnn_vitdet_b_100ep_dbias_clipart_6class.py --eval-only
CUDA_VISIBLE_DEVICES="4" python tools/lazyconfig_train_net_cbmodel.py --config-file projects/ViTDet/configs/VOC2007coco/mask_rcnn_vitdet_b_100ep_dbias_watercolor.py --eval-only
CUDA_VISIBLE_DEVICES="4" python tools/lazyconfig_train_net_cbmodel.py --config-file projects/ViTDet/configs/VOC2007coco/mask_rcnn_vitdet_b_100ep_dbias_comic.py --eval-only

### fianl eval
CUDA_VISIBLE_DEVICES="4" python tools/eval.py --config-file projects/ViTDet/configs/VOC2007coco/mask_rcnn_vitdet_b_100ep_dbias_comic.py \
    --eval-only      train.init_checkpoint=output/comic_fromw_domainbias/model_final.pth train.output_dir="output/aaa"
