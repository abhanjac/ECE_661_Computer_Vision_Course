#!/usr/bin/env python

import numpy as np, cv2, os, time, math, copy, matplotlib.pyplot as plt
from scipy import signal, optimize


#===============================================================================
# ARINDAM BHANJA CHOWDHURY
# abhanjac@purdue.edu
# ECE 661 FALL 2018, HW 7
#===============================================================================

#===============================================================================
# FUNCTIONS CREATED IN HW7.
#===============================================================================

def normalize( img, scale=255 ):
    '''
    Normalize the image and scale it to the scale value.
    '''
    img = ( img - np.amin( img ) ) / ( np.amax( img ) - np.amin( img ) )
    img = img * 255
    img = np.asarray( img, dtype=np.uint8 )
    
    return img

#===============================================================================

def LBPatPixelLoc( img=None, x=None, y=None ):
    '''
    This function finds the Local Binary Pattern (LBP) of the input gray image at 
    a pixel location given by x, y.
    '''
    if len( img.shape ) == 3:
        print( '\nERROR: Input image is not grayscale. Aborting.\n' )
        return
    
    imgH, imgW = img.shape
    
    # Return the value of the pixel themselves if they are on the boundaries.
    if x == 0 or x == imgW-1 or y == 0 or y == imgH-1:
        print( '\nERROR: Input pixels are on the boundary, cannot calculate LBPcode. ' \
                    'Aborting.\n' )
        return
    
#-------------------------------------------------------------------------------

    # Find the pattern.
    
    # Since a circular pattern is considered here, so the pixel values at the 
    # 8-neighborhood of the given pixel should be weighted by the sin and cosine
    # values of angles made by their location coordinates with the center coordinate.
    P, R = 8, 1
    theta = 2 * math.pi / P

    p0 = img[y+1, x]
    p2 = img[y, x+1]
    p4 = img[y-1, x]
    p6 = img[y, x-1]
    
    #print( p0, p2, p4, p6 )
    
    # dK is along row and dL is along col.
    p = 1
    dU, dV = R * math.cos( theta * p ), R * math.sin( theta * p )
    py, px = y + dU, x + dV
    Ay, Ax = math.floor( py ), math.floor( px )
    By, Bx = math.floor( py ), math.ceil( px )
    Cy, Cx = math.ceil( py ), math.floor( px )
    Dy, Dx = math.ceil( py ), math.ceil( px )
    A, B, C, D = img[ Ay, Ax ], img[ By, Bx ], img[ Cy, Cx ], img[ Dy, Dx ]
    dK, dL = py - math.floor( py ), px - math.floor( px )
    p1 = (1-dK) * (1-dL) * A + (1-dK) * dL * B + dK * (1-dL) * C + dK * dL * D
    
    p = 3
    dU, dV = R * math.cos( theta * p ), R * math.sin( theta * p )
    py, px = y + dU, x + dV
    Ay, Ax = math.floor( py ), math.floor( px )
    By, Bx = math.floor( py ), math.ceil( px )
    Cy, Cx = math.ceil( py ), math.floor( px )
    Dy, Dx = math.ceil( py ), math.ceil( px )
    A, B, C, D = img[ Ay, Ax ], img[ By, Bx ], img[ Cy, Cx ], img[ Dy, Dx ]
    dK, dL = py - math.floor( py ), px - math.floor( px )
    p3 = (1-dK) * (1-dL) * A + (1-dK) * dL * B + dK * (1-dL) * C + dK * dL * D
    
    p = 5
    dU, dV = R * math.cos( theta * p ), R * math.sin( theta * p )
    py, px = y + dU, x + dV
    Ay, Ax = math.floor( py ), math.floor( px )
    By, Bx = math.floor( py ), math.ceil( px )
    Cy, Cx = math.ceil( py ), math.floor( px )
    Dy, Dx = math.ceil( py ), math.ceil( px )
    A, B, C, D = img[ Ay, Ax ], img[ By, Bx ], img[ Cy, Cx ], img[ Dy, Dx ]
    dK, dL = py - math.floor( py ), px - math.floor( px )
    p5 = (1-dK) * (1-dL) * A + (1-dK) * dL * B + dK * (1-dL) * C + dK * dL * D
    
    p = 7
    dU, dV = R * math.cos( theta * p ), R * math.sin( theta * p )
    py, px = y + dU, x + dV
    Ay, Ax = math.floor( py ), math.floor( px )
    By, Bx = math.floor( py ), math.ceil( px )
    Cy, Cx = math.ceil( py ), math.floor( px )
    Dy, Dx = math.ceil( py ), math.ceil( px )
    A, B, C, D = img[ Ay, Ax ], img[ By, Bx ], img[ Cy, Cx ], img[ Dy, Dx ]
    dK, dL = py - math.floor( py ), px - math.floor( px )
    p7 = (1-dK) * (1-dL) * A + (1-dK) * dL * B + dK * (1-dL) * C + dK * dL * D
    
    pList = [ p0, p1, p2, p3, p4, p5, p6, p7 ]
    #print( pList )
    pattern = [ '0' if p < img[y, x] else '1' for p in pList ]
    patternStr = ''.join( pattern )     # Converting to string.
    LBPcode = int( patternStr, 2 )      # Converting to decimal.
    #print( type( pattern ), pattern, LBPcode )
    #print( pattern )
    
#-------------------------------------------------------------------------------
    
    # Rotate the pattern to get the least decimal value.
    
    # Instead of rotating the pattern, the same pattern in repeated twice in sequence.
    # Then this pattern is scanned by a window of 8 to find the smallest value.
    # Scanning this repeated pattern with a window of 8 in sequence is equivalent 
    # to circularly rotating the pattern.
    repeatedPattern = pattern * 2
    
    # Initializing.
    minPattern, minPatternStr, minLBPcode = pattern, patternStr, LBPcode
    
    for i in range( P ):        # P is 8 for our case.
        # Taking out a chunk of 8 consecutive elements.
        currentPattern = repeatedPattern[ i : i + P ]
        currentPatternStr = ''.join( currentPattern )   # Converting to string.
        currentLBPcode = int( currentPatternStr, 2 )    # Converting to decimal.

        if minLBPcode > currentLBPcode:     # Updating.
            minLBPcode = currentLBPcode
            minPattern = currentPattern
            minPatternStr = currentPatternStr

    #print( minLBPcode )
    #print( minPattern )
    
#-------------------------------------------------------------------------------

    # Calculating the number of sequences of 0's followed by 1's.
    
    # If the sequence is '00011011', then there are basically 2 sequences or 0's 
    # followed by 1's. So this string is split by the pattern '01' the resulting 
    # list length will give the number of sequences + 1.
    splittedPattern = minPatternStr.split('01')
    nSequences = len( splittedPattern ) - 1
    #print( splittedPattern )
    
    # The nSequences is equal to the 'number of runs' that was mentioned in lecture.
    
    # Encoding.
    encoding = P + 1        # This corresponds to the case with more than 1 sequences.
    
    if nSequences == 0:      # This is the case of all 0's (0) or all 1's (255).
        encoding = 0 if minLBPcode == 0 else P
    elif nSequences == 1:
        # In this case, sequence of 1's will appear as the second element of the list.
        # So counting the number of 1's in the second element.
        runOf1s = splittedPattern[1]
        
        # Converting the run of 1's (which is a string) into a list. So all the 
        # 1's will now become independent elements of the list, and then the length
        # of the list + 1 will give the number of 1's.
        # The +1 is because, one 1 got removed during the split by '01'.
        encoding = len( list( runOf1s ) ) + 1
    
    #print( encoding )
    
    return encoding
    
#===============================================================================

if __name__ == '__main__':
    
    # TASK 1.1
    
    # Loading the images.

    trainFilepath = './imagesDatabaseHW7/training'
    testFilepath = './imagesDatabaseHW7/testing'
    exampleFilepath = './example'
    
    classNames = [ 'beach', 'building', 'car', 'mountain', 'tree' ]
    classIdx = { cl: idx for idx, cl in enumerate( classNames ) }

##-------------------------------------------------------------------------------

    ## This is only for testing if the LBPatPixelLoc function is working properly 
    ## or not.

    ##img = np.reshape( [5,4,2,4,2,1,2,4,4], (3,3) )      # 1,1
    ##img = np.reshape( [2,4,0,0,0,2,0,2,4], (3,3) )      # 1,5
    ##img = np.reshape( [2,2,4,1,0,0,4,0,2], (3,3) )      # 1,6
    ##img = np.reshape( [2,4,2,1,2,1,4,0,4], (3,3) )      # 1,3
    #img = np.reshape( [4,2,1,2,4,4,4,1,5], (3,3) )      # 2,1
    ##img = np.reshape( [2,1,2,4,4,0,1,5,0], (3,3) )      # 2,2
    
    #print(img)
    #LBPatPixelLoc( img, 1, 1 )
        
##-------------------------------------------------------------------------------

    #img = cv2.imread( 'beach.jpg', 0 )

    #img = np.array( [ [5,4,2,4,2,2,4,0], 
                      #[4,2,1,2,1,0,0,2],
                      #[2,4,4,0,4,0,2,4],
                      #[4,1,5,0,4,0,5,5],
                      #[0,4,4,5,0,0,3,2],
                      #[2,0,4,3,0,3,1,2],
                      #[5,1,0,0,5,4,2,3],
                      #[1,0,0,4,5,5,0,1] ] )

    #imgH, imgW = img.shape
    #print( img )
    #print( imgH, imgW )

    #LBPimg = np.zeros( ( imgH, imgW ), dtype=np.uint8 )

    #for y in range( 1, imgH-1 ):        # Ignoring the boundary pixels.
        #for x in range( 1, imgW-1 ):    # Ignoring the boundary pixels.
            #print(y, x)
            #encoding = LBPatPixelLoc( img, x, y )
            #LBPimg[ y, x ] = encoding

    ## Neglecting the bounding rows and columns of the image as nothing 
    ## are encoded to those pixels. They are just 0s. But if they are not 
    ## removed, then the counts of the number of actual encoded 0s in the 
    ## histogram will be disrupted.
    ## So neglecting them before calculating the historgram.
    #LBPimg = LBPimg[ 1 : imgH-1, 1 : imgW-1 ]
    
    #P = 8
    #hist = cv2.calcHist( [LBPimg], channels=[0], mask=None, histSize=[P+2], \
                                                #ranges=[0, P+2] )
    ## Since different images can have different number of pixels, so the 
    ## histogram has to be normalized before comparison. This is done by 
    ## dividing the counts in all the bins by the total count of all bins.
    ##hist = hist / np.sum( hist ) 
    
    #hist = np.reshape( hist, (P+2) )    # Reshaping before plotting.
    
    #fig1 = plt.figure(1)
    #fig1.gca().cla()
    #plt.bar( np.arange( P+2 ), hist )  # Plot histogram.
    ##plt.show()
    
##===============================================================================

    ## Training Set.

    ## The histogram of all the training images are stored in a common list.
    ## The corresponding class index of the images are also stored in another list.
    #listOfTrainImgHist, listOfTrainImgClassIdx = [], []

    #for clIdx, cl in enumerate( classNames ):
        #trainingFolder = os.path.join( trainFilepath, cl )
        #listOfImgs = os.listdir( trainingFolder )
        
        #for idx, i in enumerate( listOfImgs ):
            
            #imgFilePath = os.path.join( trainingFolder, i )            
            #img = cv2.imread( imgFilePath, 0 )      # Read image as grayscale.

            #imgH, imgW = img.shape
            #print( f'{idx+1}: {imgFilePath}, {imgW}x{imgH}' )            

            #LBPimg = np.zeros( ( imgH, imgW ), dtype=np.uint8 )
    
            #for y in range( 1, imgH-1 ):        # Ignoring the boundary pixels.
                #for x in range( 1, imgW-1 ):    # Ignoring the boundary pixels.
                    ##print(y, x)
                    #encoding = LBPatPixelLoc( img, x, y )
                    #LBPimg[ y, x ] = encoding

            ## Neglecting the bounding rows and columns of the image as nothing 
            ## are encoded to those pixels. They are just 0s. But if they are not 
            ## removed, then the counts of the number of actual encoded 0s in the 
            ## histogram will be disrupted.
            ## So neglecting them before calculating the historgram.
            #LBPimg = LBPimg[ 1 : imgH-1, 1 : imgW-1 ]
            
            #P = 8
            #hist = cv2.calcHist( [LBPimg], channels=[0], mask=None, histSize=[P+2], \
                                                        #ranges=[0, P+2] )
            ## Since different images can have different number of pixels, so the 
            ## histogram has to be normalized before comparison. This is done by 
            ## dividing the counts in all the bins by the total count of all bins.
            #hist = hist / np.sum( hist ) 
            
            #hist = np.reshape( hist, (P+2) )    # Reshaping before plotting.
            
            #listOfTrainImgHist.append( hist )   # Storing the histogram in array.
            #listOfTrainImgClassIdx.append( classIdx[ cl ] )

    ## Converting the lists to arrays and saving them.
    #arrOfTrainImgHist = np.array( listOfTrainImgHist )
    #arrOfTrainImgClassIdx = np.array( listOfTrainImgClassIdx )

    #np.savez( 'train_hist_arrays.npz', arrOfTrainImgHist, arrOfTrainImgClassIdx )
    #print( 'File saved.' )

##===============================================================================

    ## Testing Set.

    ## The histogram of all the testing images are stored in a common list.
    ## The corresponding class index of the images are also stored in another list.
    #listOfTestImgHist, listOfTestImgClassIdx = [], []

    #testingFolder = testFilepath
    #listOfImgs = os.listdir( testingFolder )
        
    #for idx, i in enumerate( listOfImgs ):
        
        #imgFilePath = os.path.join( testingFolder, i )            
        #img = cv2.imread( imgFilePath, 0 )      # Read image as grayscale.

        #imgH, imgW = img.shape
        #print( f'{idx+1}: {imgFilePath}, {imgW}x{imgH}' )            

        #LBPimg = np.zeros( ( imgH, imgW ), dtype=np.uint8 )

        #for y in range( 1, imgH-1 ):        # Ignoring the boundary pixels.
            #for x in range( 1, imgW-1 ):    # Ignoring the boundary pixels.
                ##print(y, x)
                #encoding = LBPatPixelLoc( img, x, y )
                #LBPimg[ y, x ] = encoding

        ## Neglecting the bounding rows and columns of the image as nothing 
        ## are encoded to those pixels. They are just 0s. But if they are not 
        ## removed, then the counts of the number of actual encoded 0s in the 
        ## histogram will be disrupted.
        ## So neglecting them before calculating the historgram.
        #LBPimg = LBPimg[ 1 : imgH-1, 1 : imgW-1 ]
        
        #P = 8
        #hist = cv2.calcHist( [LBPimg], channels=[0], mask=None, histSize=[P+2], \
                                                    #ranges=[0, P+2] )
        ## Since different images can have different number of pixels, so the 
        ## histogram has to be normalized before comparison. This is done by 
        ## dividing the counts in all the bins by the total count of all bins.
        #hist = hist / np.sum( hist ) 
        
        #hist = np.reshape( hist, (P+2) )    # Reshaping before plotting.
        
        #listOfTestImgHist.append( hist )   # Storing the histogram in array.
        
        #cl = i.split('_')[0]    # This is the class label of the test image.
        
        #listOfTestImgClassIdx.append( classIdx[ cl ] )
        
        #if i == 'car_2.jpg' or i == 'beach_4.jpg' or i == 'building_2.jpg' or \
           #i == 'mountain_3.jpg' or i == 'tree_2.jpg':
            #fig1 = plt.figure(1)
            #fig1.gca().cla()
            #plt.bar( np.arange( P+2 ), hist )  # Plot histogram.
            #plt.title( f'LBP Histogram for a {cl} image' )
            ##plt.show()
            #fig1.savefig( f'LBP_Histogram_{cl}_image.png' )
            
            #LBPimg = normalize( LBPimg )
            ##cv2.imshow( 'LBPimg', LBPimg )
            ##cv2.waitKey(0)
            #cv2.imwrite( f'LBP_image_{cl}.png', LBPimg )

    ## Converting the lists to arrays and saving them.
    #arrOfTestImgHist = np.array( listOfTestImgHist )
    #arrOfTestImgClassIdx = np.array( listOfTestImgClassIdx )

    #np.savez( 'test_hist_arrays.npz', arrOfTestImgHist, arrOfTestImgClassIdx )
    #print( 'File saved.' )

##===============================================================================

    # Nearest Neighbor Classification.

    #npzFile = np.load( 'example_hist_arrays.npz' )
    npzFile = np.load( 'train_hist_arrays.npz' )
    
    arrOfTrainImgHist = npzFile[ 'arr_0' ]
    
    # Removing last bin as it consists of unwanted features.
    arrOfTrainImgHist = arrOfTrainImgHist[ :, :-1 ]
    
    arrOfTrainImgClassIdx = npzFile[ 'arr_1' ]    

    print( arrOfTrainImgHist.shape )
    print( arrOfTrainImgClassIdx.shape )

    #npzFile = np.load( 'example_hist_arrays.npz' )
    npzFile = np.load( 'test_hist_arrays.npz' )

    arrOfTestImgHist = npzFile[ 'arr_0' ]

    # Removing last bin as it consists of unwanted features.
    arrOfTestImgHist = arrOfTestImgHist[ :, :-1 ]

    arrOfTestImgClassIdx = npzFile[ 'arr_1' ]    

    print( arrOfTestImgHist.shape )
    print( arrOfTestImgClassIdx.shape )


    nTrainImgs = arrOfTrainImgHist.shape[0]
    nTestImgs = arrOfTestImgHist.shape[0]
    
    accuracy = 0
    
    for i in range( nTestImgs ):
    #for i in range( 1 ):
        testHist = arrOfTestImgHist[i]
        testClassIdx = arrOfTestImgClassIdx[i]
        
        # Repeating the test hist and the corresponding test class idx as many 
        # number of times as there are training images. This is done for ease of
        # subtraction.
        testHistArray = np.ones( ( nTrainImgs, 1 ) ) * testHist
        
        distArray = arrOfTrainImgHist - testHistArray
        distNormArray = np.linalg.norm( distArray, axis=1 )
        
        sortedIndex = np.argsort( distNormArray )   # Sorted indices.
        sortedDist = np.sort( distNormArray )       # Sorted values.

        # The shortest distance will correspond to the predicted class distance.
        # So the index (in the trainClassIdxArray) corresponding to this shortest 
        # distance will give the predicted class.
        
        # Now we have to find the class corresponding to the least 5 distances.
        # The one which appears the most among these 5 choices, will be considered
        # the predicted class.
        predictionList = [0,0,0,0,0]
        
        # K for Nearest Neighbor classifier.
        K = 7
        
        for k in range( K ):
            predClassIdx = arrOfTrainImgClassIdx[ sortedIndex[ k ] ]
            # Increment the element of the predictionList corresponding to the 
            # current predClassIdx.
            predictionList[ predClassIdx ] += 1
        
        #print( predictionList, '  ', end='' )      # Match pattern.
        
        predClassIdx = predictionList.index( max( predictionList ) )
        #predClassIdx = arrOfTrainImgClassIdx[ sortedIndex[ 0 ] ]
        
        print( f'{i+1}, True: {testClassIdx}, Predicted: {predClassIdx}' )
                
        if predClassIdx == testClassIdx:        accuracy += 1
        
    accuracy = accuracy * 100 / nTestImgs
    
    print( f'Test Accuracy: {accuracy} %' )
    
