FOLDS_ROOT = 'mskf_folds/'
PROJECT = 'HuBMAP'
MODEL_V = 'yolov8x-seg'
PREFIX = 'mskf_'
# path_to_data = f'{FOLDS_ROOT}/fold{i}/hubmap-coco.yaml'
print(CFG())
print()

oof = []
folds_i = [0, 1, 2, 3, 4]
for i in folds_i:
    models = [
        YOLO(f'{PROJECT}/{PREFIX}{MODEL_V}-fold{i}/weights/best.pt'), 
#         YOLO(f'{PROJECT}/{PREFIX}{MODEL_V}-fold{i}/weights/last.pt'),
             ]


    val_dir = f'{FOLDS_ROOT}/fold{i}/valid/'
    val_ids = [r.split('.')[0] for r in os.listdir(val_dir) if r.endswith('.txt')][:35]

    annotations = []
    pred_masks = []
    pred_scores = []
    for image_id in val_ids:
        annotations.append(parse_ann(id_to_annotation[image_id]))
        image = Image.open(os.path.join(val_dir ,image_id+'.tif'))

        raw_masks, raw_boxes, raw_scores = predict(image=image,
                                models=models,
                                transforms=CFG.transforms,
                                imgsz=CFG.imgsz,
                                conf=CFG.conf,
                                iou_nms=CFG.iou_nms,
                                retina_masks=CFG.retina_masks,
                               )
        
        if len(raw_boxes):
            if CFG.method == 'wbf':
                masks, boxes, scores = combine_results_wbf(
                    raw_masks, raw_boxes, raw_scores,
                    iou_thr=CFG.iou_nms_ensemble, min_votes=3
                )
            elif CFG.method == 'nms':
                masks, boxes, scores = combine_results_nmms(
                    raw_masks, raw_boxes,
                    raw_scores, iou_nms=CFG.iou_nms_ensemble)
            else:
                masks, boxes, scores = raw_masks, raw_boxes, raw_scores
        else:
            masks, boxes, scores = raw_masks, raw_boxes, raw_scores
        
        if len(boxes):
            boxes = np.concatenate((boxes, scores[:, None]), axis=1) # add conf to boxes
            masks, boxes  = postprocess_masks(
                masks,
                boxes,
                conf_thresh=CFG.conf_thresh,
                min_size=CFG.min_size,
                dilation_n_iter=CFG.dilation_n_iter,
                remove_overlap=CFG.remove_overlap,
                corrupt=CFG.corrupt,
            )
            scores = boxes[:, 4]
            
        else:
            masks = np.zeros((1, 512, 512), dtype=np.uint8)
            scores = np.array([0])
        
        pred_masks.append(masks)
        pred_scores.append(scores)

    ious = evaluate_model(annotations, pred_masks, pred_scores)
    print(f'fold {i}')
    print(f'IOU mean: {np.mean(ious):.4}')
    print(f'IOU std: {np.std(ious):.4}')
    print()
    result_df = pd.DataFrame(ious, columns=['iou'], index=val_ids)
    result_df['source_wsi'] = df.loc[result_df.index.values]['source_wsi']
    oof.append(result_df)
oof = pd.concat(oof)
oof['source_wsi'] = oof['source_wsi'].astype('category')
for wsi, subdf in oof.groupby('source_wsi'):
    print(f'wsi {wsi}')
    print(f'IOU mean: {np.mean(subdf.iou.values):.4}')
    print(f'IOU std: {np.std(subdf.iou.values):.4}')
    
print(f'overall')
print(f'IOU mean: {np.mean(oof.iou.values):.4}')
print(f'IOU std: {np.std(oof.iou.values):.4}')
