python e2e_inference_pro.py \
--config 'configuration/maskrcnn_resnext101_DCN_160e_icdar.py' \
--det_weights '../resnext101_DCN_160e_epoch_150.pth' \
--rec_weights '../transformerocr_btc.pth' \
--root '../Dataset/MapT12' \
--input_images '../Dataset/MapT12/2.NotcontainingTSHS/NotVietnamese' \
--output_destination '../Dataset/dev/2.NotcontainingTSHS/NotVietnamese' 