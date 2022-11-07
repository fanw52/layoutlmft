import json
import os

from transformers import AutoTokenizer

from layoutlmft.data.utils import load_image, merge_bbox, normalize_bbox, simplify_bbox

tokenizer = AutoTokenizer.from_pretrained("/data/wufan/algos/data/transformers_data/XFUN/model/layoutxlm-base")


def generate_examples(filepaths):
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for doc in data["documents"]:
            doc["img"]["fpath"] = os.path.join(filepath[:-5], doc["img"]["fname"])
            image, size = load_image(doc["img"]["fpath"])
            document = doc["document"]
            tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
            entities = []
            relations = []
            id2label = {}
            entity_id_to_index_map = {}
            empty_entity = set()
            for line in document:
                if len(line["text"]) == 0:
                    empty_entity.add(line["id"])
                    continue
                id2label[line["id"]] = line["label"]
                relations.extend([tuple(sorted(l)) for l in line["linking"]])
                tokenized_inputs = tokenizer(
                    line["text"],
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    return_attention_mask=False,
                )
                text_length = 0
                ocr_length = 0
                bbox = []
                last_box = None
                for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
                    if token_id == 6:
                        bbox.append(None)
                        continue
                    text_length += offset[1] - offset[0]
                    tmp_box = []
                    while ocr_length < text_length:
                        ocr_word = line["words"].pop(0)
                        ocr_length += len(
                            tokenizer._tokenizer.normalizer.normalize_str(ocr_word["text"].strip())
                        )
                        tmp_box.append(simplify_bbox(ocr_word["box"]))
                    if len(tmp_box) == 0:
                        tmp_box = last_box
                    bbox.append(normalize_bbox(merge_bbox(tmp_box), size))
                    last_box = tmp_box
                bbox = [[bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b for i, b in
                        enumerate(bbox)]
                if line["label"] == "other":
                    label = ["O"] * len(bbox)
                else:
                    label = [f"I-{line['label'].upper()}"] * len(bbox)
                    label[0] = f"B-{line['label'].upper()}"
                tokenized_inputs.update({"bbox": bbox, "labels": label})
                if label[0] != "O":
                    entity_id_to_index_map[line["id"]] = len(entities)
                    entities.append(
                        {
                            "start": len(tokenized_doc["input_ids"]),
                            "end": len(tokenized_doc["input_ids"]) + len(tokenized_inputs["input_ids"]),
                            "label": line["label"].upper(),
                        }
                    )
                for i in tokenized_doc:
                    tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]
            relations = list(set(relations))
            relations = [rel for rel in relations if rel[0] not in empty_entity and rel[1] not in empty_entity]
            kvrelations = []
            for rel in relations:
                pair = [id2label[rel[0]], id2label[rel[1]]]
                if pair == ["question", "answer"]:
                    kvrelations.append(
                        {"head": entity_id_to_index_map[rel[0]], "tail": entity_id_to_index_map[rel[1]]}
                    )
                elif pair == ["answer", "question"]:
                    kvrelations.append(
                        {"head": entity_id_to_index_map[rel[1]], "tail": entity_id_to_index_map[rel[0]]}
                    )
                else:
                    continue

            def get_relation_span(rel):
                bound = []
                for entity_index in [rel["head"], rel["tail"]]:
                    bound.append(entities[entity_index]["start"])
                    bound.append(entities[entity_index]["end"])
                return min(bound), max(bound)

            relations = sorted(
                [
                    {
                        "head": rel["head"],
                        "tail": rel["tail"],
                        "start_index": get_relation_span(rel)[0],
                        "end_index": get_relation_span(rel)[1],
                    }
                    for rel in kvrelations
                ],
                key=lambda x: x["head"],
            )
            chunk_size = 512
            for chunk_id, index in enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size)):
                item = {}
                for k in tokenized_doc:
                    item[k] = tokenized_doc[k][index: index + chunk_size]
                entities_in_this_span = []
                global_to_local_map = {}
                for entity_id, entity in enumerate(entities):
                    if (
                            index <= entity["start"] < index + chunk_size
                            and index <= entity["end"] < index + chunk_size
                    ):
                        entity["start"] = entity["start"] - index
                        entity["end"] = entity["end"] - index
                        global_to_local_map[entity_id] = len(entities_in_this_span)
                        entities_in_this_span.append(entity)
                relations_in_this_span = []
                for relation in relations:
                    if (
                            index <= relation["start_index"] < index + chunk_size
                            and index <= relation["end_index"] < index + chunk_size
                    ):
                        relations_in_this_span.append(
                            {
                                "head": global_to_local_map[relation["head"]],
                                "tail": global_to_local_map[relation["tail"]],
                                "start_index": relation["start_index"] - index,
                                "end_index": relation["end_index"] - index,
                            }
                        )
                item.update(
                    {
                        "id": f"{doc['id']}_{chunk_id}",
                        "image": image,
                        "entities": entities_in_this_span,
                        "relations": relations_in_this_span,
                    }
                )
                print(f"{doc['id']}_{chunk_id}", item)


if __name__ == '__main__':
    filepath = ["/data/wufan/algos/data/transformers_data/XFUN/data/zh.train.json"]

    generate_examples(filepath)

# import json
#
# path = "/data/wufan/algos/data/transformers_data/XFUN/data/zh.val.json"
# output_path = "/data/wufan/algos/data/transformers_data/XFUN/data/zh.val.new.json"
# result = {}
# with open(path) as rfile:
#     data = json.load(rfile)  # ["documents"]
#     result["lang"] = data["lang"]
#     result["version"] = data["version"]
#     result["split"] = data["split"]
#     result["documents"] = []
#     documents = data["documents"]
#     new_documents = []
#     for document in documents:
#         print(document)
#         for line in document["document"]:
#             box = line['box']
#             words = line["words"]
#             for word in words:
#                 word["box"] = box
#         new_documents.append(document)
#     result["documents"] = new_documents
#
#
# # with open(output_path,'w') as writer:
# #     json.dump(result,writer,ensure_ascii=False)

'''
解析文书数据
1、 数据xfun中，每个实体单独用一个框标注，而在文书数据中，一行数据中可能包含多个实体，且不能提供每一个字符的位置，因此，为了方便起见
我们用整行文本的位置替代每个字符的位置
2、在


'''