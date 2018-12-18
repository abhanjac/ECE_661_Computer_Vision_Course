#!/usr/bin/env python

import numpy as np, cv2, os, time, math, copy, matplotlib.pyplot as plt
from scipy import signal, optimize


#===============================================================================
# ARINDAM BHANJA CHOWDHURY
# abhanjac@purdue.edu
# ECE 661 FALL 2018, HW 6
#===============================================================================

#===============================================================================
# FUNCTIONS CREATED IN HW6.
#===============================================================================

def applyOtsu( img=None, kRange=[0, 256] ):
    '''
    Takes in a single channel image and finds out the threshold value k using 
    otsu algorithm. Also applies the k to filter the image and returns a binary
    filtered image.
    The kRange specifies the range of bins of the histogram that will be used to 
    find the threshold.
    '''    
    hist = cv2.calcHist( [img], channels=[0], mask=None, histSize=[256], ranges=[0, 256] )
    # The [0] means calculate the hist on the 0th channel.

    # Only take the rows of hist that falls in the range of kRange.
    # Basically this can be used to truncate the histogram.
    hist = hist[ kRange[0] : kRange[1], : ]

    # Finding the probability distribution.
    N = np.sum( hist )     # Total number of pixels in the histogram.
    P = hist / N
    PT = np.sum(P)     # Sum of all probability. This theoritically should be 1, 
    # but because of the floating point division may be like 9.999999, or even 1.00001.
    # So if it is 1.00001, then subtracting the sum of all probability (when k will 
    # be in the last bin) from 1, will make the count -ve. So PT will be used instead
    # of 1.

#-------------------------------------------------------------------------------
    
    # Bins can have different ranges. Not necessarily from 0 to 256.
    # If bins are like 100 to 150. So bin values will be like: [100, 101, 102, ... 149, 150].
    bins = np.mgrid[ kRange[0] : kRange[1] ]    # Bin values.
    #print(bins.shape)
    
    muT = np.matmul( bins, P )
    #print(N, P.shape, muT.shape, muT, np.sum(img)/N)
    
    varBlist = []   # List of between class variances.
    for idx, k in enumerate( bins ):
        wk = np.sum( P[ : idx+1 ] )
        muk = np.matmul( bins[ : idx+1 ], P[ : idx+1 ] )
        
        # Calculating the variances.
        varB = ( muT * wk - muk ) * ( muT * wk - muk ) / ( wk * ( PT - wk ) + 0.000001 )
        varBlist.append( varB )
        # The 0.000001 is there to avoid division by 0 (which will happen in case 
        # when k pointing to the end of the last bin so wk will be equal to PT then
        # which means all the counts in the entire histogram is considered, and so 
        # the dinominator will be 0).
        # In some cases, if the first bin also does not have any pixels, then wk
        # may be 0. The 0.00001 saves that case as well.
        
        #print(wk, varB)

#-------------------------------------------------------------------------------

    varBarray = np.array( varBlist )    # Array of between class variances.
       
    # Calculate the threshold value (k).
    maxVarB = np.amax( varBarray, axis=0 )      # Max varB (between class variance).
    maxVarBidx = np.argmax( varBarray, axis=0 )
    bestK = bins[ maxVarBidx ]      # Index of the max varB in the varBarray is best k. 
    
    #print(bestK, maxVarB)
    
    return bestK[0]     # bestK is an array of 1 element. So returning 0th element of that.

#===============================================================================

def segmentByOtsu( img, kRanges=[[0, 256], [0, 256], [0, 256]] ):
    '''
    This function takes in an image and applies otsu segmentation to the image.
    The kRanges specifies the range of bins of the histogram that will be used to 
    find the threshold in the different channels.
    '''    
    # Splitting the images and making a list of the channels.
    if len(img.shape) == 2:     # Gray image.
        imgChannels = [img]
        h, w = img.shape
        c = 1
    elif len(img.shape) == 3:   # Colored image.
        b, g, r = cv2.split(img)
        imgChannels = [b, g, r]
        h, w, c = img.shape
    else:
        print( '\nERROR: Image is not gray or colored. Aborting.\n' )
        return
    
#-------------------------------------------------------------------------------

    # Calculate the histogram of the channels.
    fig1 = plt.figure(1)
    fig1.gca().cla()        # Clear the axes for new plots.
    color = ['b', 'g', 'r']
    kList, filteredImgList = [], []
    combinedImg = np.ones( (h, w), dtype=np.uint8 ) * 255

    for idx, ch in enumerate( imgChannels ):
        # Blurring the image a little bit for segmentation to work better.
        ch = cv2.GaussianBlur( ch, (5,5), 0 )
        
        kRange = kRanges[ idx ]
        
        hist = cv2.calcHist( [ch], channels=[0], mask=None, histSize=[256], ranges=[0,256] )
        # The [0] means calculate the hist on the 0th channel.
        
        # Only take the rows of hist that falls in the range of kRange.
        # Basically this can be used to truncate the histogram.
        hist = hist[ kRange[0] : kRange[1], : ]
        
        plt.plot( hist, color=color[idx] )  # Plot histogram.
        
        # Apply otsu algorithm to the current channel.
        k = applyOtsu( ch, kRange=kRange )
        kList.append( k )
        
        plt.plot( [k,k], [0,100], color[idx], linewidth=2.0 )     # Show k on histogram.
        
        # Create the binary filtered image by using the k value.
        _, filteredImg = cv2.threshold( ch, k, 255, cv2.THRESH_BINARY )

        filteredImgList.append( filteredImg )
        
        plt.xlabel('bins'); plt.ylabel('no. of pixels'); plt.title('Histogram'); plt.grid()
    
    #plt.show()
    #fig1.savefig( f'./hist_2.png' )
    
    # Return the list of k's and filteredImgs.
    return kList, filteredImgList
    
#===============================================================================

def calcVarImg( img=None, kernel=None ):
    '''
    This function takes in an image and a kernel size and calculates a variance 
    image which is just an array containing the variance values calculated from a
    neighborhood of the size of the given kernel around every pixel of given image.
    The input kernel has the format (h, w).
    '''
    kernelH, kernelW = kernel[0], kernel[1]
    if len(img.shape) == 1:     # Image is grayscale.
        imgH, imgW = img.shape
    else:       # Image is colored, then convert to grayscale.
        img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        imgH, imgW = img.shape

    kernelForMU = np.ones( kernel ) / ( kernelH * kernelW )
        
    meanImg = signal.convolve2d( img, kernelForMU, mode='same' )
    
    # Convert the img from uint8 to float image so that when doing the square of 
    # the image the values do not round off to 256.
    imgSquare = np.asarray( img, dtype=np.float32 ) * np.asarray( img, dtype=np.float32 )
    
    #print( np.amin(imgSquare), np.amax(imgSquare) )
    
    meanImgSquareImg = signal.convolve2d( imgSquare, kernelForMU, mode='same' )
    
    varianceImg = meanImgSquareImg - meanImg * meanImg
    
    #print( np.amin( varianceImg ), np.amax( varianceImg ) )
    
    return varianceImg

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

def findContour( mask ):
    '''
    This function takes in a binary image and finds the contour that separate the 
    foreground and the background in the image.
    '''
    maskH, maskW = mask.shape
    contour = np.zeros( mask.shape, dtype=np.uint8 )
    
    N = 3       # 3x3 window, with 8-neighbors.
    halfSize = int( N / 2 )

    # Only if the center pixel value is 255, and not all of its surrounding pixels
    # equals to 255, is considered as a valid contour point between the foreground 
    # and background.

    for x in range( halfSize, maskW-halfSize ):
        for y in range( halfSize, maskH-halfSize ):
            window = mask[ y - halfSize : y + halfSize + 1, 
                           x - halfSize : x + halfSize + 1 ]
            #print( window.shape )
            if window[1, 1] == 255 and np.sum( window ) < int( N * N * 255 ):
                contour[y, x] = 255

    return contour

#===============================================================================

if __name__ == '__main__':
    
    # TASK 1.1
    
    # Loading the images.

    filepath = './HW6Pics'

#-------------------------------------------------------------------------------

    # Color based segmentation for lighthouse image.
    filename = '1.jpg'
    img = cv2.imread( os.path.join( filepath, filename ) )
    kList, filteredImgList = segmentByOtsu( img, kRanges=[[0, 256], [0, 256], [0, 256]] )
    
    # Red is the dominant color in the lighthouse image. Hence the red mask is 
    # anded with the compliment of blue and green masks to get a segmented lighthouse.
    # This observation is done after looking at the original image and the filtered
    # image returned by segmentByOtsu function.

    bChan, gChan, rChan = filteredImgList[0], filteredImgList[1], filteredImgList[2]
    
    cv2.imwrite( f'./bchannel_{filename[:-4]}.png', bChan )
    cv2.imwrite( f'./gchannel_{filename[:-4]}.png', gChan )
    cv2.imwrite( f'./rchannel_{filename[:-4]}.png', rChan )
    
#-------------------------------------------------------------------------------

    combinedImg = rChan   # Starting with red.
    
    bChanInv = cv2.bitwise_not( bChan )  # Complimenting blue.

    # Anding with the compliment of blue.
    combinedImg = cv2.bitwise_and( combinedImg, bChanInv )
    
    gChanInv = cv2.bitwise_not( gChan )  # Complimenting green.

    # Anding with the compliment of green.
    combinedImg = cv2.bitwise_and( combinedImg, gChanInv )

    channels = np.hstack( ( bChan, gChan, rChan ) )
    cv2.imshow( 'Channels', channels )
    cv2.imshow( 'Combined Image', combinedImg )
    cv2.waitKey(0)
    cv2.imwrite( f'./combined_image_{filename[:-4]}.png', combinedImg )
    
#===============================================================================

    # Color based segmentation for ski image.
    filename = '3.jpg'
    img = cv2.imread( os.path.join( filepath, filename ) )
    kList, filteredImgList = segmentByOtsu( img, kRanges=[[0, 256], [0, 256], [0, 256]] )
    
    bChan, gChan, rChan = filteredImgList[0], filteredImgList[1], filteredImgList[2]
    
    cv2.imwrite( f'./bchannel_{filename[:-4]}.png', bChan )
    cv2.imwrite( f'./gchannel_{filename[:-4]}.png', gChan )
    cv2.imwrite( f'./rchannel_{filename[:-4]}.png', rChan )

#-------------------------------------------------------------------------------

    # NEED ITERATION 2.

    # The foreground is filtered out in the blue image. So this means the foreground
    # actually lies below the k threshold. So inverting the bChan to get the
    # region that contains the foreground.
    # This observation is done after looking at the original image and the filtered
    # image returned by segmentByOtsu function.
    bChanInv = cv2.bitwise_not( bChan )  # Complimenting blue.
    
    img2 = img[:,:,0]

    kList2, filteredImgList2 = segmentByOtsu( img2, kRanges=[[0, kList[0]]] )    
    bChan2 = filteredImgList2[0]
    
    cv2.imwrite( f'./bchannel_iter_2_{filename[:-4]}.png', bChan2 )
    
#-------------------------------------------------------------------------------

    # Inverting the bChan again to get the foreground.
    bChan2Inv = cv2.bitwise_not( bChan2 )  # Complimenting blue.

    # There are some missing parts in this image, which were visible in the red 
    # channel. So this is or-ed with the red to get a better contour. This is 
    # a temporary image.
    # But still the part of the snow. So the green and red channel was and-ed
    # and then this xor-ed with the temporary image.
    # This observation is done after looking at the original image and the filtered
    # image returned by segmentByOtsu function.
    
    # Oring with the red channel of iteration 1.
    combinedImg = cv2.bitwise_or( bChan2Inv, rChan )
    
    # Oring green and red channel of iteration 1.
    GandR = cv2.bitwise_and( gChan, rChan )
    
    # Xoring with the green channel of iteration 1.
    combinedImg = cv2.bitwise_xor( combinedImg, GandR )
    
    cv2.imshow( 'Combined Image', combinedImg )
    cv2.waitKey(0)
    cv2.imwrite( f'./combined_image_{filename[:-4]}.png', combinedImg )
    
#===============================================================================

    # Color based segmentation for baby image.
    filename = '2.jpg'
    img = cv2.imread( os.path.join( filepath, filename ) )
    kList, filteredImgList = segmentByOtsu( img, kRanges=[[0, 256], [0, 256], [0, 256]] )
    
    bChan, gChan, rChan = filteredImgList[0], filteredImgList[1], filteredImgList[2]
    
    cv2.imwrite( f'./bchannel_{filename[:-4]}.png', bChan )
    cv2.imwrite( f'./gchannel_{filename[:-4]}.png', gChan )
    cv2.imwrite( f'./rchannel_{filename[:-4]}.png', rChan )

#-------------------------------------------------------------------------------

    # The baby is mostly white. So the red, blue and green are present in equal 
    # amounts. The images obtained are inverted and or-ed together to fill up
    # some of the black patches.
    bChanInv = cv2.bitwise_not( bChan )  # Complimenting blue.
    gChanInv = cv2.bitwise_not( gChan )  # Complimenting blue.
    rChanInv = cv2.bitwise_not( rChan )  # Complimenting blue.

    combinedImg = rChanInv   # Starting with red.
    
    # Oring with the compliment of green.
    combinedImg = cv2.bitwise_or( combinedImg, gChanInv )
    combinedImg = cv2.bitwise_or( combinedImg, bChanInv )

    channels = np.hstack( ( bChan, gChan, rChan ) )
    cv2.imshow( 'Channels', channels )
    cv2.imshow( 'Combined Image', combinedImg )
    cv2.waitKey(0)
    cv2.imwrite( f'./combined_image_{filename[:-4]}.png', combinedImg )

#===============================================================================

    # Texture based segmentation for lighthouse image.
    
    filename = '1.jpg'
    img = cv2.imread( os.path.join( filepath, filename ) )

    kernelH, kernelW = 3, 3
    varianceImg3x3 = calcVarImg( img, (kernelH, kernelH) )
    
    cv2.imshow( 'Variance Image', varianceImg3x3 )
    cv2.imwrite( f'./texture_image_{filename[:-4]}_{kernelH}x{kernelW}.png', varianceImg3x3 )
    
    # Since varianceImg3x3 has values which are very large than 255 because of the 
    # square terms involved. So they have to be normalized and converted to a range 
    # of 0 to 255.
    
    normalizedImg3x3 = normalize( varianceImg3x3 )
    cv2.imshow( f'Normalized Image {kernelH}x{kernelW}', normalizedImg3x3 )
    cv2.imwrite( f'./normalized_texture_image_{filename[:-4]}_{kernelH}x{kernelW}.png', normalizedImg3x3 )
    cv2.waitKey(0)
    
#-------------------------------------------------------------------------------

    kernelH, kernelW = 5, 5
    varianceImg5x5 = calcVarImg( img, (kernelH, kernelH) )
    
    cv2.imshow( 'Variance Image', varianceImg5x5 )
    cv2.imwrite( f'./texture_image_{filename[:-4]}_{kernelH}x{kernelW}.png', varianceImg5x5 )
    
    # Since varianceImg5x5 has values which are very large than 255 because of the 
    # square terms involved. So they have to be normalized and converted to a range 
    # of 0 to 255.
    
    normalizedImg5x5 = normalize( varianceImg5x5 )
    cv2.imshow( f'Normalized Image {kernelH}x{kernelW}', normalizedImg5x5 )
    cv2.imwrite( f'./normalized_texture_image_{filename[:-4]}_{kernelH}x{kernelW}.png', normalizedImg5x5 )
    cv2.waitKey(0)
    
#-------------------------------------------------------------------------------

    kernelH, kernelW = 7, 7
    varianceImg7x7 = calcVarImg( img, (kernelH, kernelH) )
    
    cv2.imshow( 'Variance Image', varianceImg7x7 )
    cv2.imwrite( f'./texture_image_{filename[:-4]}_{kernelH}x{kernelW}.png', varianceImg7x7 )
    
    # Since varianceImg7x7 has values which are very large than 255 because of the 
    # square terms involved. So they have to be normalized and converted to a range 
    # of 0 to 255.
    
    normalizedImg7x7 = normalize( varianceImg7x7 )
    cv2.imshow( f'Normalized Image {kernelH}x{kernelW}', normalizedImg7x7 )
    cv2.imwrite( f'./normalized_texture_image_{filename[:-4]}_{kernelH}x{kernelW}.png', normalizedImg7x7 )
    cv2.waitKey(0)

    normalizedImg = cv2.merge( ( normalizedImg3x3, normalizedImg5x5, normalizedImg7x7 ) )
    
    cv2.imshow( 'Normalized Image Combined', normalizedImg )
    cv2.imwrite( f'./combined_normalized_texture_image_{filename[:-4]}.png', normalizedImg )
    cv2.waitKey(0)
    
#-------------------------------------------------------------------------------

    # Applying otsu to find the segments from the normalized combined texture images.

    originalNormImg = copy.deepcopy( normalizedImg )

    nIter = 3       # Number of iterations for otsu.
    bchan = normalizedImg3x3
    gchan = normalizedImg5x5
    rchan = normalizedImg7x7
    kList=[256, 256, 256]
    
    for i in range( nIter ):
        normalizedImg = cv2.merge( ( bchan, gchan, rchan ) )    # Merging.
        cv2.imshow( 'normalizedImg', normalizedImg )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        kRanges = [ [ 0, kList[0] ], [ 0, kList[1] ], [ 0, kList[2] ] ]
        #kRanges = [ [ kList[0], 256 ], [ kList[1], 256 ], [ kList[2], 256 ] ]

        kList, filteredImgList = segmentByOtsu( normalizedImg, kRanges=kRanges )
        
        bChan, gChan, rChan = filteredImgList[0], filteredImgList[1], filteredImgList[2]
        
        print( kList )
        
        cv2.imwrite( f'./bchannel_{filename[:-4]}.png', bChan )
        cv2.imwrite( f'./gchannel_{filename[:-4]}.png', gChan )
        cv2.imwrite( f'./rchannel_{filename[:-4]}.png', rChan )
        #channels = np.hstack( ( bChan, gChan, rChan ) )
        #cv2.imshow( 'Channels', channels )
        
        andedChannel = cv2.bitwise_or( bChan, gChan )
        andedChannel = cv2.bitwise_or( andedChannel, rChan )
        cv2.imshow( 'Combined Anded Channel', andedChannel )
        cv2.imwrite( f'./final_ORED_texture_image_{filename[:-4]}.png', andedChannel )
        cv2.waitKey(0)

#===============================================================================

    # Finding the contours.
    
    # The mask image created earlier is read in.
    maskFileName = 'combined_image_3.png'
    #maskFileName = 'final_ORED_texture_image_3.png'
    mask = cv2.imread( maskFileName )
    
    # The mask is 3 channel but it is created by boolean operations, hence it has
    # values as only 0,0,0 and 255,255,255.

    # Converting it into single channel, as the findContour only takes in single
    # channel images.
    mask = cv2.cvtColor( mask, cv2.COLOR_BGR2GRAY )

    cv2.imshow( 'Mask', mask )
    cv2.waitKey(0)
    contour = findContour( mask )
    cv2.imshow( 'Contour', contour )
    cv2.waitKey(0)
    cv2.imwrite( f'./{maskFileName[:-4]}_contour.png', contour )



