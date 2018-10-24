import tensorflow as tf
import sys
import os
import csv

def classify_image(image_path, headers):
    f = open('submit.csv','w')
    writer = csv.DictWriter(f, fieldnames = headers)
    writer.writeheader()
    
    
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]
   
    #obtained the pretrained graph
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    files = os.listdir(image_path)
    with tf.Session() as sess:
         for file in files:
             # Read the image_data
                image_data = tf.gfile.FastGFile(image_path+'/'+file, 'rb').read()
                # Feed the image_data as input to the graph and get first prediction
                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

                predictions = sess.run(softmax_tensor, \
                                       {'DecodeJpeg/contents:0': image_data})

                # Sort to show labels of first prediction in order of confidence
                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                records = []
                row_dict = {}
                head, tail = os.path.split(file)
                row_dict['id'] = tail.split('.')[0]

                for node_id in top_k:
                    human_string = label_lines[node_id]
                    human_string = human_string.replace(" ","_")
                    print("node_id=", node_id)
                    score = predictions[0][node_id]
                    if(human_string == 'german_short_haired_pointer'):
                        human_string = 'german_short-haired_pointer'
                    if(human_string == 'shih_tzu'):
                        human_string = 'shih-tzu'
                    if(human_string == 'wire_haired_fox_terrier'):
                        human_string = 'wire-haired_fox_terrier'
                    if(human_string == 'curly_coated_retriever'):
                        human_string = 'curly-coated_retriever'
                    if(human_string == 'black_and_tan_coonhound'):
                        human_string = 'black-and-tan_coonhound'
                    if(human_string == 'soft_coated_wheaten_terrier'):
                        human_string = 'soft-coated_wheaten_terrier'  
                    if(human_string == 'flat_coated_retriever'):
                    	human_string = 'flat-coated_retriever' 
                    
                    print('%s (score = %.5f)' % (human_string, score))
                    row_dict[human_string] = score
                print("--------  Next Image ----------------")
                records.append(row_dict.copy())
                writer.writerows(records)
    f.close()    

def main():
    test_data_folder = 'test'
    
    template_file = open('sample_submission.csv','r')
    d_reader = csv.DictReader(template_file)

    headers = d_reader.fieldnames
    template_file.close()
    classify_image(test_data_folder, headers)
    

if __name__ == '__main__':
    main()

