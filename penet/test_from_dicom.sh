python test_from_dicom.py --input_study  ./study/bec1b1d73f48 \
                          --series_description CTPA \
                          --ckpt_path penet_best.pth.tar \
                          --device cuda \
                          --gpu_ids 0
