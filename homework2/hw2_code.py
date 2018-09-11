#!/usr/bin/env python

import numpy as np, cv2, os, time

#===============================================================================
# ARINDAM BHANJA CHOWDHURY
# abhanjac@purdue.edu
# ECE 661 FALL 2018, HW 2
#===============================================================================

#===============================================================================
# FUNCTIONS CREATED IN HW2.
#===============================================================================

# Global variables that will mark the points in the image by mouse click.
ix, iy = -1, -1

#===============================================================================

def markPoints( event, x, y, flags, params ):
    '''
    This is a function that is called on mouse callback.
    '''
    global ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y

#===============================================================================

def selectPts( filePath=None ):
    '''
    This function opens the image and lets user select the points in it.
    These points are returned as a list.
    If the image is bigger than 640 x 480, it is displayed as 640 x 480. But
    the points are mapped and stored as per the original dimension of the image.
    The points are clicked by mouse on the image itself and they are stored in
    the listOfPts.
    '''
    
    global ix, iy
    
    img = cv2.imread( filePath )
    h, w = img.shape[0], img.shape[1]
    
    w1, h1, wRatio, hRatio, resized = w, h, 1, 1, False
    print( 'Image size: {}x{}'.format(w, h) )
    
    if w > 640:
        w1, resized = 640, True
        wRatio = w / w1
    if h > 480:
        h1, resized = 480, True
        hRatio = h / h1

    if resized:     img = cv2.resize( img, (w1, h1), \
                                        interpolation=cv2.INTER_AREA )

    cv2.namedWindow( 'Image' )
    cv2.setMouseCallback( 'Image', markPoints )  # Function to detect mouseclick
    key = ord('`')

#-------------------------------------------------------------------------------
    
    listOfPts = []      # List to collect the selected points.
    
    while key & 0xFF != 27:         # Press esc to break.

        imgTemp = np.array( img )      # Temporary image.

        # Displaying all the points in listOfPts on the image.
        for i in range( len(listOfPts) ):
            cv2.circle( imgTemp, tuple(listOfPts[i]), 3, (0, 255, 0), -1 )
            
        # After clicking on the image, press any key (other than esc) to display
        # the point on the image.
        
        if ix > 0 and iy > 0:    
            print( 'New point: ({}, {}). Press \'s\' to save.'.format(ix, iy) )
            cv2.circle( imgTemp, (ix, iy), 3, (0, 0, 255), -1 )
            # Since this point is not saved yet, so it is displayed on the 
            # temporary image and not on actual img1.
        
        cv2.imshow( 'Image', imgTemp )
        key = cv2.waitKey(0)
        
        # If 's' is pressed then the point is saved to the listOfPts.
        if key == ord('s'):
            listOfPts.append( [ix, iy] )
            cv2.circle( imgTemp, (ix, iy), 3, (0, 255, 0), -1 )
            img1 = imgTemp
            ix, iy = -1, -1
            print( 'Point Saved.' )
            
        elif key == ord('d'):   ix, iy = -1, -1  # Delete point by pressing 'd'.
        
    # Map the selected points back to the size of original image using the 
    # wRatio and hRatio (if they we resized earlier).
    if resized:   listOfPts = [ [ int( p[0] * wRatio ), int( p[1] * hRatio ) ] \
                                                        for p in listOfPts ]
    
    return listOfPts

#===============================================================================

# Converts an input point (x, y) into homogeneous format (x, y, 1).
homogeneousFormat = lambda pt: [ pt[0], pt[1], 1 ]

#===============================================================================

# Converts an input point in homogeneous coordinate (x, y, z) into planar form.
planarFormat = lambda pt: [ int(pt[0] / pt[2]), int(pt[1] / pt[2]) ]

#===============================================================================

# Converts an input point in homogeneous coordinate (x, y, z) into planar form,
# But does not round them into integers, keeps them as floats.
planarFormatFloat = lambda pt: [ pt[0] / pt[2], pt[1] / pt[2] ]

#===============================================================================

def homography( srcPts=None, dstPts=None ):
    '''
    The srcPts and dstPts are the list of points from which the 3x3 homography 
    matrix (H) is to be decuced. The equation will be dstPts = H * srcPts.
    The srcPts and dstPts should be a list of at least 8 points (each point is 
    in the homogeneous coordinate form, arranged as a sublist of 3 elements 
    [x, y, 1]).
    The homography matrix is 3x3 but since only the ratios matter in homography,
    so one of the element can be taken as 1 and so there are only 8 unknowns.
    Here the last element of H is considered to be 1.
    '''
    
    # Number of source and destination points should be the same and should have
    # a one to one correspondence.
    if len(srcPts) != len(dstPts):
        print( 'srcPts and dstPts have different number of points. Aborting.' )
        return
    
    nPts = len( srcPts )
    
    # Creating matrix A and X (for the equation A * h = X).
    A, X = [], []
    for i in range( nPts ):
        # With each pair of points from srcPts and dstPts, 2 rows of A are made.
        xs, ys, xd, yd = srcPts[i][0], srcPts[i][1], dstPts[i][0], dstPts[i][1]
        row1 = [ xs, ys, 1, 0,  0,  0, -xs*xd, -ys*xd ]
        row2 = [ 0,  0,  0, xs, ys, 1, -xs*yd, -ys*yd ]
        A.append( row1 )
        A.append( row2 )
        X.append( xd )
        X.append( yd )
    #print( A )
    #print( X )
    
    # Converting A into nPts x nPts array and X into a nPts x 1 array.
    A, X = np.array( A ), np.reshape( X, ( nPts*2, 1 ) )
    Ainv = np.linalg.inv( A )
    h = np.matmul( Ainv, X )        # A * h = X, so h = Ainv * X.
    #print(h)
    
    # Appending a 1 for last element
    h = np.insert( h, nPts*2, 1 )
    H = np.reshape( h, (3,3) )      # Reshaping to 3x3.
    
    return H

#===============================================================================

# The input is a 2 element list of the homography matrix H and the point pt.
# [H, pt]. Applies H to the point pt. pt is in homogeneous coordinate form.
applyHomography = lambda HandPt: np.matmul( HandPt[0], \
                                            np.reshape( HandPt[1], (3,1) ) )

#===============================================================================

# Functions to find the min and max x and y coordinates from a list of points.
maxX = lambda listOfPts: sorted( listOfPts, key=lambda x: x[0] )[-1][0]
minX = lambda listOfPts: sorted( listOfPts, key=lambda x: x[0] )[0][0]
maxY = lambda listOfPts: sorted( listOfPts, key=lambda x: x[1] )[-1][1]
minY = lambda listOfPts: sorted( listOfPts, key=lambda x: x[1] )[0][1]

#===============================================================================

def rectifyColor( pt=None, img=None ):
    '''
    This function takes in a point which is in float (not int) and an image and
    gives out a weighted average of the color of the surrounding pixels to 
    which the point may be mapped to when converted from float to int.
    It is meant for those points which gets mapped to subpixel locations instead
    of mapping perfectly in an integer location in the given img.
    
    Format of pt is (x, y), like opencv. 
    Returns the values as a numpy array.
    '''
    x1, y1 = np.floor( pt[0] ), np.floor( pt[1] )
    x2, y2 = np.ceil( pt[0] ),  np.floor( pt[1] )
    x3, y3 = np.floor( pt[0] ), np.ceil( pt[1] )
    x4, y4 = np.ceil( pt[0] ),  np.ceil( pt[1] )

    x, y = int( pt[0] ), int( pt[1] )   # Location where pt is to be mapped.
    
    # Distances of the potential location from the final location (x, y).
    # The + 0.0000001 is to prevent division by 0.
    dX1 = np.linalg.norm( np.array( [ x-x1, y-y1 ] ) ) + 0.0000001
    dX2 = np.linalg.norm( np.array( [ x-x2, y-y2 ] ) ) + 0.0000001
    dX3 = np.linalg.norm( np.array( [ x-x3, y-y3 ] ) ) + 0.0000001
    dX4 = np.linalg.norm( np.array( [ x-x4, y-y4 ] ) ) + 0.0000001
    
#-------------------------------------------------------------------------------

    h, w = img.shape[0], img.shape[1]
    
    #print( x1, y1, x2, y2, x3, y3, x4, y4 )
    
    # This is done so that while taking int(), the x, y dont go out of bound.
    x1 = int(x1) if int(x1) < w else w-1
    x2 = int(x2) if int(x2) < w else w-1
    x3 = int(x3) if int(x3) < w else w-1
    x4 = int(x4) if int(x4) < w else w-1

    y1 = int(y1) if int(y1) < h else h-1
    y2 = int(y2) if int(y2) < h else h-1
    y3 = int(y3) if int(y3) < h else h-1
    y4 = int(y4) if int(y4) < h else h-1
    
    # Color values at the above locations.
    C1, C2, C3, C4 = img[ int(y1) ][ int(x1) ], img[ int(y2) ][ int(x2) ], \
                     img[ int(y3) ][ int(x3) ], img[ int(y4) ][ int(x4) ]
    
#-------------------------------------------------------------------------------

    # Final color. So the farther the potential location is from the final 
    # location, more is the corresponding distance, and hence less will be the
    # effect of the color of that location on the final color. In the above 
    # calculations basically the following equation is evaluated.
    C = ( (C1 / dX1) + (C2 / dX2) + (C3 / dX3) + (C4 / dX4) ) / ( \
                     (1 / dX1) + (1 / dX2) + (1 / dX3) + (1 / dX4) )
    
    C = np.asarray( C, dtype=np.uint8 )
    
    return C

#===============================================================================

def mapImages( sourceImg=None, targetImg=None, pqrs=None, H=None ):
    '''
    This function takes in a source image, a target image and the four 
    corner points (pqrs) of the target that defines the region where the source
    image is to be projected. It also takes in homography matrix from the target
    to the source image. Returns the projected image.

    pqrs should be a list of points. Each point in the list is a sublist of x 
    and y coordinate of the point. These should be in planar (not homogeneous)
    coordinate.
    '''
    
    # Drawing a black polygon to make all the pixels in the target region = 0,
    # before mapping.
    targetImg = cv2.fillPoly( targetImg, np.array( [ pqrs ] ), (0,0,0) )
    
    targetH, targetW = targetImg.shape[0], targetImg.shape[1]
    sourceH, sourceW = sourceImg.shape[0], sourceImg.shape[1]
    
    processingTime = time.time()
    
    # Mapping the points.
    # Scanning only the region between the points p, q, r, s.
    for r in range( minY( pqrs ), maxY( pqrs ) ):
        for c in range( minX( pqrs ), maxX( pqrs ) ):

            if targetImg[r][c].all() == 0:
                # If point is in the 0 polygon.
                # which indicates that the point is in the black polygon where
                # the source image is to be projected.
                targetPt = homogeneousFormat( [ c, r ] )
                sourcePt = planarFormatFloat( applyHomography( [H, targetPt] ) )
                
                if sourcePt[0] < sourceW and sourcePt[1] < sourceH and \
                   sourcePt[0] > 0 and sourcePt[1] > 0:
                       
                    # Mapping the source point pixel to the target point pixel.
                    targetImg[r][c] = sourceImg[ int( sourcePt[1] ) ][ \
                                                        int( sourcePt[0] ) ]
                    #targetImg[r][c] = rectifyColor( pt=sourcePt, img=sourceImg )
                    pass

    print( 'Time taken: {}'.format( time.time() - processingTime ) )
    
    return targetImg    # Returning target image after mapping target into it.

#===============================================================================

if __name__ == '__main__':
    
    # TASK 1a.
    
    filePath = './PicsHw2'
    filename1, filename2, filename3 = '1.jpg', '2.jpg', '3.jpg'
    faceFileName = 'Jackie.jpg'
    
    # Reading the points.
    
    #pqrsFig1 = selectPts( filePath=os.path.join( filePath, filename1 ) )
    pqrsFig1 = [[1518, 180], [2940, 728], [2996, 2031], [1495, 2240]]
    #print( 'Points of {}: {}\n'.format( filename1, pqrsFig1 ) )
    
    #pqrsFig2 = selectPts( filePath=os.path.join( filePath, filename2 ) )
    pqrsFig2 = [[1331, 344], [3002, 626], [3025, 1896], [1309, 2009]]
    #print( 'Points of {}: {}\n'.format( filename2, pqrsFig2 ) )
    
    #pqrsFig3 = selectPts( filePath=os.path.join( filePath, filename3 ) )
    pqrsFig3 = [[931, 744], [2793, 395], [2855, 2223], [903, 2093]]
    #print( 'Points of {}: {}\n'.format( filename3, pqrsFig3 ) )
    
    #pqrsFace = selectPts( filePath=os.path.join( filePath, faceFileName ) )
    pqrsFace = [[0, 0], [1280, 0], [1280, 720], [0, 720]]
    #print( 'Points of {}: {}\n'.format( faceFileName, pqrsFace ) )    
    
#-------------------------------------------------------------------------------

    # Converting points to homogeneous coordinates.
    
    pqrsHomFmt1 = [ homogeneousFormat( pt ) for pt in pqrsFig1 ]
    pqrsHomFmt2 = [ homogeneousFormat( pt ) for pt in pqrsFig2 ]
    pqrsHomFmt3 = [ homogeneousFormat( pt ) for pt in pqrsFig3 ]
    pqrsHomFmtF = [ homogeneousFormat( pt ) for pt in pqrsFace ]
    
#-------------------------------------------------------------------------------

    # Finding the homography.
    
    # Homography between target and fig1.
    Hbetw1ToFace = homography( srcPts=pqrsHomFmt1, dstPts=pqrsHomFmtF )
    #print( 'Homography between {} -> {}: \n{}\n'.format( filename1, \
                                            #faceFileName, Hbetw1ToFace ) )

    # Homography between target and fig2.
    Hbetw2ToFace = homography( srcPts=pqrsHomFmt2, dstPts=pqrsHomFmtF )
    #print( 'Homography between {} -> {}: \n{}\n'.format( filename2, \
                                            #faceFileName, Hbetw2ToFace ) )

    # Homography between target and fig3.
    Hbetw3ToFace = homography( srcPts=pqrsHomFmt3, dstPts=pqrsHomFmtF )
    #print( 'Homography between {} -> {}: \n{}\n'.format( filename3, \
                                            #faceFileName, Hbetw3ToFace ) )

#-------------------------------------------------------------------------------

    # Implanting the target into figs.
    
    # Reading the images.
    sourceImg = cv2.imread( os.path.join( filePath, faceFileName ) )
    targetImg = cv2.imread( os.path.join( filePath, filename1 ) )
    
    targetImg = mapImages( sourceImg=sourceImg, targetImg=targetImg, \
                                            pqrs=pqrsFig1, H=Hbetw1ToFace )
    
    # Resizing for the purpose of display.
    targetImg = cv2.resize( targetImg, (640,480), interpolation=cv2.INTER_AREA )
    cv2.imshow( 'Mapped Image', targetImg )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Reading the images.
    sourceImg = cv2.imread( os.path.join( filePath, faceFileName ) )
    targetImg = cv2.imread( os.path.join( filePath, filename2 ) )
    
    targetImg = mapImages( sourceImg=sourceImg, targetImg=targetImg, \
                                            pqrs=pqrsFig2, H=Hbetw2ToFace )
    
    # Resizing for the purpose of display.
    targetImg = cv2.resize( targetImg, (640,480), interpolation=cv2.INTER_AREA )
    cv2.imshow( 'Mapped Image', targetImg )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Reading the images.
    sourceImg = cv2.imread( os.path.join( filePath, faceFileName ) )
    targetImg = cv2.imread( os.path.join( filePath, filename3 ) )
    
    targetImg = mapImages( sourceImg=sourceImg, targetImg=targetImg, \
                                            pqrs=pqrsFig3, H=Hbetw3ToFace )
    
    # Resizing for the purpose of display.
    targetImg = cv2.resize( targetImg, (640,480), interpolation=cv2.INTER_AREA )
    cv2.imshow( 'Mapped Image', targetImg )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#-------------------------------------------------------------------------------

    # TASK 1b.
    
    # Applying the product of H from 1 to 2 and H from 2 to 3 to the fig1.

    Hbetw2To1 = homography( srcPts=pqrsHomFmt2, dstPts=pqrsHomFmt1 )
    Hbetw3To2 = homography( srcPts=pqrsHomFmt3, dstPts=pqrsHomFmt2 )
    
    Hbetw3To1 = homography( srcPts=pqrsHomFmt3, dstPts=pqrsHomFmt1 )
    print( 'H by direct homography calculation: \n{}'.format( Hbetw3To1 ) )

    Hbetw3To1 = np.matmul( Hbetw2To1, Hbetw3To2 )    
    print( 'H by product of to homographies (applied to image): \n{}'.format( \
                                                                Hbetw3To1 ) )
    # Reading the images.
    sourceImg = cv2.imread( os.path.join( filePath, filename1 ) )
    targetImg = np.zeros( sourceImg.shape )
    tgtH, tgtW = targetImg.shape[0], targetImg.shape[1]
    H = Hbetw3To1
    
    processingTime = time.time()
    
    for r in range( tgtH ):
        for c in range( tgtW ):
            targetPt = homogeneousFormat( [ c, r ] )
            targetPt = applyHomography( [H, targetPt] )
            
            targetPt = planarFormat( targetPt )
            #print( r, c )
            
            if targetPt[0] > 0 and targetPt[0] < tgtW and \
               targetPt[1] > 0 and targetPt[1] < tgtH:
                targetImg[r][c] = sourceImg[ targetPt[1] ][ targetPt[0] ]
            
    print( 'Time taken: {}'.format( time.time() - processingTime ) )
    
    # Resizing for the purpose of display.
    targetImg = np.asarray( targetImg, dtype=np.uint8 )
    targetImg = cv2.resize( targetImg, (640,480), interpolation=cv2.INTER_AREA )
    cv2.imshow( 'Transformed Image', targetImg )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#===============================================================================

    # TASK 2.
    
    filePath = './PicsSelf'
    filename1, filename2, filename3 = '1.jpg', '2.jpg', '3.jpg'
    faceFileName = 'self.jpg'
    
    # Reading the points.
    
    #pqrsFig1 = selectPts( filePath=os.path.join( filePath, filename1 ) )
    pqrsFig1 = [[616, 584], [1864, 399], [1875, 1777], [853, 2065]]
    #print( 'Points of {}: {}\n'.format( filename1, pqrsFig1 ) )
    
    #pqrsFig2 = selectPts( filePath=os.path.join( filePath, filename2 ) )
    pqrsFig2 = [[711, 716], [1791, 545], [1714, 1797], [824, 2104]]
    #print( 'Points of {}: {}\n'.format( filename2, pqrsFig2 ) )
    
    #pqrsFig3 = selectPts( filePath=os.path.join( filePath, filename3 ) )
    pqrsFig3 = [[809, 326], [1353, 623], [1203, 1855], [773, 1388]]
    #print( 'Points of {}: {}\n'.format( filename3, pqrsFig3 ) )

    #pqrsFace = selectPts( filePath=os.path.join( filePath, faceFileName ) )
    pqrsFace = [[5, 15], [1188, 12], [1186, 1478], [9, 1481]]
    #print( 'Points of {}: {}\n'.format( faceFileName, pqrsFace ) )    
    
#-------------------------------------------------------------------------------

    # Converting points to homogeneous coordinates.
    
    pqrsHomFmt1 = [ homogeneousFormat( pt ) for pt in pqrsFig1 ]
    pqrsHomFmt2 = [ homogeneousFormat( pt ) for pt in pqrsFig2 ]
    pqrsHomFmt3 = [ homogeneousFormat( pt ) for pt in pqrsFig3 ]
    pqrsHomFmtF = [ homogeneousFormat( pt ) for pt in pqrsFace ]
    
#-------------------------------------------------------------------------------

    # Finding the homography.
    
    # Homography between target and fig1.
    Hbetw1ToFace = homography( srcPts=pqrsHomFmt1, dstPts=pqrsHomFmtF )
    #print( 'Homography between {} -> {}: \n{}\n'.format( filename1, \
                                            #faceFileName, Hbetw1ToFace ) )

    # Homography between target and fig2.
    Hbetw2ToFace = homography( srcPts=pqrsHomFmt2, dstPts=pqrsHomFmtF )
    #print( 'Homography between {} -> {}: \n{}\n'.format( filename2, \
                                            #faceFileName, Hbetw2ToFace ) )

    # Homography between target and fig3.
    Hbetw3ToFace = homography( srcPts=pqrsHomFmt3, dstPts=pqrsHomFmtF )
    #print( 'Homography between {} -> {}: \n{}\n'.format( filename3, \
                                            #faceFileName, Hbetw3ToFace ) )

#-------------------------------------------------------------------------------

    # Implanting the target into figs.
    
    # Reading the images.
    sourceImg = cv2.imread( os.path.join( filePath, faceFileName ) )
    targetImg = cv2.imread( os.path.join( filePath, filename1 ) )
    
    targetImg = mapImages( sourceImg=sourceImg, targetImg=targetImg, \
                                            pqrs=pqrsFig1, H=Hbetw1ToFace )
    
    # Resizing for the purpose of display.
    targetImg = cv2.resize( targetImg, (640,480), interpolation=cv2.INTER_AREA )
    cv2.imshow( 'Mapped Image', targetImg )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Reading the images.
    sourceImg = cv2.imread( os.path.join( filePath, faceFileName ) )
    targetImg = cv2.imread( os.path.join( filePath, filename2 ) )
    
    targetImg = mapImages( sourceImg=sourceImg, targetImg=targetImg, \
                                            pqrs=pqrsFig2, H=Hbetw2ToFace )
    
    # Resizing for the purpose of display.
    targetImg = cv2.resize( targetImg, (640,480), interpolation=cv2.INTER_AREA )
    cv2.imshow( 'Mapped Image', targetImg )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Reading the images.
    sourceImg = cv2.imread( os.path.join( filePath, faceFileName ) )
    targetImg = cv2.imread( os.path.join( filePath, filename3 ) )
    
    targetImg = mapImages( sourceImg=sourceImg, targetImg=targetImg, \
                                            pqrs=pqrsFig3, H=Hbetw3ToFace )
    
    # Resizing for the purpose of display.
    targetImg = cv2.resize( targetImg, (640,480), interpolation=cv2.INTER_AREA )
    cv2.imshow( 'Mapped Image', targetImg )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#-------------------------------------------------------------------------------

    # Applying the product of H from 1 to 2 and H from 2 to 3 to the fig1.
    
    Hbetw2To1 = homography( srcPts=pqrsHomFmt2, dstPts=pqrsHomFmt1 )
    Hbetw3To2 = homography( srcPts=pqrsHomFmt3, dstPts=pqrsHomFmt2 )
    
    Hbetw3To1 = homography( srcPts=pqrsHomFmt3, dstPts=pqrsHomFmt1 )
    print( 'H by direct homography calculation: \n{}'.format( Hbetw3To1 ) )

    Hbetw3To1 = np.matmul( Hbetw2To1, Hbetw3To2 )    
    print( 'H by product of to homographies (applied to image): \n{}'.format( \
                                                                Hbetw3To1 ) )
    # Reading the images.
    sourceImg = cv2.imread( os.path.join( filePath, filename1 ) )
    targetImg = np.zeros( sourceImg.shape )
    tgtH, tgtW = targetImg.shape[0], targetImg.shape[1]
    H = Hbetw3To1
    
    processingTime = time.time()
    
    for r in range( tgtH ):
        for c in range( tgtW ):
            targetPt = homogeneousFormat( [ c, r ] )
            targetPt = applyHomography( [H, targetPt] )
            
            targetPt = planarFormat( targetPt )
            print( r, c )
            
            if targetPt[0] > 0 and targetPt[0] < tgtW and \
               targetPt[1] > 0 and targetPt[1] < tgtH:
                targetImg[r][c] = sourceImg[ targetPt[1] ][ targetPt[0] ]
            
    print( 'Time taken: {}'.format( time.time() - processingTime ) )
    
    # Resizing for the purpose of display.
    targetImg = np.asarray( targetImg, dtype=np.uint8 )
    targetImg = cv2.resize( targetImg, (640,480), interpolation=cv2.INTER_AREA )
    cv2.imshow( 'Transformed Image', targetImg )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#===============================================================================

    ## Extra images, not included in hw.

    #filename4, filename5 = '4.jpg', '5.jpg'

    ##pqrsFig4 = selectPts( filePath=os.path.join( filePath, filename4 ) )
    #pqrsFig4 = [[325, 851], [1012, 569], [1720, 1203], [960, 1653]]
    ##print( 'Points of {}: {}\n'.format( filename4, pqrsFig4 ) )

    ##pqrsFig5 = selectPts( filePath=os.path.join( filePath, filename5 ) )
    #pqrsFig5 = [[465, 821], [1108, 205], [1898, 1218], [1207, 1666]]
    ##print( 'Points of {}: {}\n'.format( filename5, pqrsFig5 ) )


    #pqrsHomFmt4 = [ homogeneousFormat( pt ) for pt in pqrsFig4 ]
    #pqrsHomFmt5 = [ homogeneousFormat( pt ) for pt in pqrsFig5 ]


    ## Homography between target and fig3.
    #Hbetw4ToFace = homography( srcPts=pqrsHomFmt4, dstPts=pqrsHomFmtF )
    ##print( 'Homography between {} -> {}: \n{}\n'.format( filename4, \
                                            ##faceFileName, Hbetw4ToFace ) )

    ## Homography between target and fig3.
    #Hbetw5ToFace = homography( srcPts=pqrsHomFmt5, dstPts=pqrsHomFmtF )
    ##print( 'Homography between {} -> {}: \n{}\n'.format( filename5, \
                                            ##faceFileName, Hbetw5ToFace ) )


    ## Reading the images.
    #sourceImg = cv2.imread( os.path.join( filePath, faceFileName ) )
    #targetImg = cv2.imread( os.path.join( filePath, filename4 ) )
    
    #targetImg = mapImages( sourceImg=sourceImg, targetImg=targetImg, \
                                            #pqrs=pqrsFig4, H=Hbetw4ToFace )
    
    ## Resizing for the purpose of display.
    #targetImg = cv2.resize( targetImg, (640,480), interpolation=cv2.INTER_AREA )
    #cv2.imshow( 'Mapped Image', targetImg )
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    ## Reading the images.
    #sourceImg = cv2.imread( os.path.join( filePath, faceFileName ) )
    #targetImg = cv2.imread( os.path.join( filePath, filename5 ) )
    
    #targetImg = mapImages( sourceImg=sourceImg, targetImg=targetImg, \
                                            #pqrs=pqrsFig5, H=Hbetw5ToFace )
    
    ## Resizing for the purpose of display.
    #targetImg = cv2.resize( targetImg, (640,480), interpolation=cv2.INTER_AREA )
    #cv2.imshow( 'Mapped Image', targetImg )
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

