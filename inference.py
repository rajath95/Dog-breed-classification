import tensorflow as tf
import sys
import os
import csv


def classify_image(filename, headers):
    f = open('sample_submit.csv','w')
    writer = csv.DictWriter(f, fieldnames = headers)
    writer.writeheader()
    
    
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]
   
    #obtained the pretrained graph
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


    image_path=os.getcwd()
    files=[1]	
    with tf.Session() as sess:
         for file in files:
             # Read the image_data
                image_data = tf.gfile.FastGFile(image_path+'/'+filename, 'rb').read()
                # Feed the image_data as input to the graph and get first prediction
                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

                predictions = sess.run(softmax_tensor, \
                                       {'DecodeJpeg/contents:0': image_data})

                # Sort to show labels of first prediction in order of confidence
                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                records = []
                row_dict = {}
                #head, tail = os.path.split(file)
                row_dict['id'] = filename.split('.')[0]
                
                

                for node_id in top_k[:1]:
                #    print("node_id=", node_id)
                    human_string = label_lines[node_id]
                    human_string = human_string.replace(" ","_")
                    score = predictions[0][node_id]
                    print('Prediction: %s (score = %.5f)' % (human_string, score))
                    row_dict[human_string] = score
                
                records.append(row_dict.copy())
                writer.writerows(records)
    f.close()    

def main():
	
    filename=sys.argv[1]
    template_file = open('sample_submission.csv','r')
    d_reader = csv.DictReader(template_file)

    headers = d_reader.fieldnames
    template_file.close()
    classify_image(filename, headers)
    

if __name__ == '__main__':
    main()
   
    

