
import sys
import csv
import uuid
import argparse 
import math
 
def SigmoidNeuralNetwork():
    args = parser.parse_args()
    file = args.data
    learning_rate = float(args.eta)
    iterations =float(args.iterations)
    #file = 'C://Users//manis//Desktop//neuralnetworkassignment//Gauss3.csv'
    #learning_rate = float(0.2)
    #iterations = 2   
    #with open( 'C://Users//manis//Desktop//neuralnetworkassignment//Gauss3.csv' ,mode =  'r') as file:
    with open( file ,  'r') as file:
        csvFile = csv.reader( file , delimiter= ',' ) 
        data=[row for row in csvFile]
        #print(data)
        a = [ ] 
        b = [ ] 
        #initializing list for y values
        t = [ ] 
        #finding number of rows in our dataset
        rows = len(data) 
        #  print(rows)
        #finding number of columns in our dataset
        columns = len(data[0])
        #  print(columns)
        counter = 0
        #seperating dependent and independent variables 
        for i in range(0 , rows) :
            a1 = [ ]
            b1 = [ ]
            dependent_variable = [ ]
            counter= 0      
            for j in data[i] :          
                counter += 1
                if counter == columns :       
                    dependent_variable.append(float(j))
                elif counter == columns - 1 :
                    b1.append(float(j))  
                else:
                    a1.append(float(j))
            t.append(dependent_variable)        
            a.append(a1)
            b.append(b1)
            #initialising weights
        w_bias_h1 =  0.2
        w_a_h1 =    -0.3
        w_b_h1 =     0.4
        w_bias_h2 = -0.5
        w_a_h2 =    -0.1
        w_b_h2 =    -0.4
        w_bias_h3 =  0.3
        w_a_h3 =     0.2
        w_b_h3 =     0.1
        w_bias_o =  -0.1
        w_h1_o =     0.1
        w_h2_o =     0.3
        w_h3_o =    -0.4
        #initialising
        h1 = 0.0
        h2 = 0.0
        h3 = 0.0
        o = 0.0
        delta_o = 0.0
        delta_h1 = 0.0
        delta_h2 = 0.0
        delta_h3 = 0.0
        #starting learning
        count = 1

        print('a,b,h1,h2,h3,o,t,delta_h1,delta_h2,delta_h3,delta_o,w_bias_h1,w_a_h1,w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o,w_h3_o')
        print('-,-,-,-,-,-,-,-,-,-,-,0.20000,-0.30000,0.40000,-0.50000,-0.10000,-0.40000,0.30000,0.20000,0.10000,-0.10000,0.10000,0.30000,-0.4')                       
        while(iterations > 0 ):
            iterations = iterations - 1                 
            for i in range( 0 , len(a) ): 
                print("{0:.5f}".format(a[i][0]) , end =',')
                print("{0:.5f}".format(b[i][0]) , end =',')
                #summation at h1
                h1 = (w_bias_h1) + ( w_a_h1 * a[i][0] ) + (w_b_h1 * b[i][0] )
                #sigmoid function for h1
                h1 = float(1 / (1 + math.exp(-h1)))
                #printing output
                print("{0:.5f}".format(h1) , end =',')
                #summation at h2
                h2 = (w_bias_h2) + ( w_a_h2 * a[i][0] ) + (w_b_h2 * b[i][0] )
                #sigmoid function for h2
                h2 = float(1 / (1 + math.exp(-h2)))
                #printing output
                print("{0:.5f}".format(h2) , end =',')
                #summation at h3
                h3 = (w_bias_h3) + ( w_a_h3 * a[i][0] ) + (w_b_h3 * b[i][0] )
                #sigmoid function for h3
                h3 = float(1 / (1 + math.exp(-h3)))
                print("{0:.5f}".format(h3) , end =',')
                #at output layer 
                #summation at output layer o 
                o = (w_bias_o) + ( w_h1_o * h1 ) + (w_h2_o * h2 ) + (w_h3_o * h3 ) 
                #sigmoid function for output layer o
                o = float(1 / (1 + math.exp(-o)))
                #printing output       
                print( "{0:.5f}".format(o) , end =',' )
                #printing output of our dataset 
                print("{0:.5f}".format(round(t[i][0] , 5)) , end =',' )
                #calculating error in backpropogation
                delta_o = ( (o) * (1-o) * (t[i][0] - o)) 
                delta_h1 = ( (h1) * (1-h1) * (delta_o * w_h1_o))
                print("{0:.5f}".format(delta_h1) , end =',')
                delta_h2 = ( (h2) * (1-h2) * (delta_o * w_h2_o))
                print("{0:.5f}".format(delta_h2) , end =',' )
                delta_h3 = ( (h3) * (1-h3) * (delta_o * w_h3_o))
                print("{0:.5f}".format(delta_h3) , end =',' )
                print("{0:.5f}".format(delta_o) , end =',' )
                #weight updation @h1
                w_bias_h1 = (w_bias_h1 + ((learning_rate) * delta_h1 )) 
                print("{0:.5f}".format(w_bias_h1) , end =',')
                w_a_h1 = (w_a_h1 + ((learning_rate) * delta_h1 * a[i][0] )) 
                print("{0:.5f}".format(w_a_h1) , end =',' )
                w_b_h1 = (w_b_h1 + ((learning_rate) * delta_h1 * b[i][0] ))
                print("{0:.5f}".format(w_b_h1) , end =',' )
                #weight updation @h2
                w_bias_h2 = (w_bias_h2 + ((learning_rate) * delta_h2 ))
                print("{0:.5f}".format(w_bias_h2 ) , end =',')
                w_a_h2 = (w_a_h2 + ((learning_rate) * delta_h2 * a[i][0] ))
                print("{0:.5f}".format(w_a_h2) , end =',')
                w_b_h2 = (w_b_h2 + ((learning_rate) * delta_h2 * b[i][0] ))
                print("{0:.5f}".format(w_b_h2) , end =',' )
                #weight updation @h3
                w_bias_h3 = (w_bias_h3 + ((learning_rate) * delta_h3 ))
                print("{0:.5f}".format(w_bias_h3) , end =',' )
                w_a_h3 = (w_a_h3 + ((learning_rate) * delta_h3 * a[i][0] )) 
                print("{0:.5f}".format(w_a_h3) , end =',' )
                w_b_h3 = (w_b_h3 + ((learning_rate) * delta_h3 * b[i][0] )) 
                print("{0:.5f}".format(w_b_h3) , end =',' )
                #weight updation @o
                w_bias_o = (w_bias_o + ((learning_rate) * delta_o )) 
                print("{0:.5f}".format(w_bias_o) , end =',' )
                w_h1_o = (w_h1_o + ((learning_rate) * delta_o * h1 )) 
                print("{0:.5f}".format(w_h1_o) , end =',' )
                w_h2_o = (w_h2_o + ((learning_rate) * delta_o * h2 )) 
                print("{0:.5f}".format(w_h2_o) , end =',')
                w_h3_o = (w_h3_o + ((learning_rate) * delta_o * h3 ))
                print("{0:.5f}".format(w_h3_o) )
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Data File")
    parser.add_argument("-l", "--eta", help="Learning Rate")
    parser.add_argument("-t", "--iterations", help="iterations")
    SigmoidNeuralNetwork()













        
        
        