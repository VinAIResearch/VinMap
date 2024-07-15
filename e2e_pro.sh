python e2e_inference_pro.py \
--config 'configuration/maskrcnn_resnext101_DCN_160e_icdar.py' \
--det_weights '../resnext101_DCN_160e_epoch_150.pth' \
--rec_weights '../transformerocr_btc.pth' \
--root '../Dataset/Vietnam_map' \
--input_images '../Dataset/Vietnam_map_develop/negative/vietnamese' \
--output_destination '../Dataset/Vietnam_map/dev/negative/vietnamese' 