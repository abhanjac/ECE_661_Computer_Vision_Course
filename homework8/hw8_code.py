#!/usr/bin/env python

import numpy as np, cv2, os, time, math, copy, matplotlib.pyplot as plt
from scipy import signal, optimize


#===============================================================================
# ARINDAM BHANJA CHOWDHURY
# abhanjac@purdue.edu
# ECE 661 FALL 2018, HW 8
#===============================================================================

#===============================================================================
# FUNCTIONS CREATED IN HW8.
#===============================================================================

def rodriguesRtoW( R ):
    '''
    Converting R to vector form w.
    '''
    # Before taking the arccos of a value, care should be taken to check whether
    # that value is less than 1 or not.
    phi = math.acos( ( np.trace(R) - 1 ) * 0.5  )    
    
    w = ( 0.5 * phi / math.sin( phi ) ) * np.array( [ [ R[2,1]-R[1,2] ], \
                                                      [ R[0,2]-R[2,0] ], \
                                                      [ R[1,0]-R[0,1] ] ] )
    return w
        
#===============================================================================

def rodriguesWtoR( w ):
    '''
    Converting w to matrix form R.
    '''
    # Before taking the arccos of a value, care should be taken to check whether
    # that value is less than 1 or not.
    phi = np.linalg.norm( w )
    
    # Converting w to skew symmetrix form.
    wx = np.array( [ [ 0, -w[2], w[1] ], [ w[2], 0, -w[0] ], [ -w[1], w[0], 0 ] ] )
    
    R = np.eye(3) + ( math.sin( phi ) / phi ) * wx + \
                ( ( 1 - math.cos( phi ) ) / ( phi * phi ) ) * np.matmul( wx, wx )
    
    return R
        
#===============================================================================

def refineParamByLM( setOfXij, XWij, K, setOfR, setOfT, k1=None, k2=None ):
    '''
    This function takes in the Xij location of the corners in all the patterns, 
    and also the XWij world locations of those corners, 
    (The Xij, XW are in planar coordinates. So the XW vector will be 
    equal to [xw, yw, 0] transpose (as zw is considered 0 for all the patterns.
    But before calculating the error, those have to be converted to homogeneous 
    form) along with R, K, t and then calculates the error between the true Xij and
    the XijEst esimated using the calibration parameters. Then the XijEst (is 
    converted back to planar form and) is compared to the Xij and the error is 
    calculated. This error is minimized using Levenberg-Marqdth algorithm.
    k1 and k2 are the radial distortion parameters. So if they are not None, then 
    that means some valid initialized values are given for these. Hence they are
    also optimized.
    '''
    XWij = [ [ pt[0], pt[1], 0, 1 ] for pt in XWij ]     # Convert to homogeneous form.
    # zw is 0 for all the corners as the pattern is assumed to be in z=0 plane.
    
    alphax, s, x0, alphay, y0 = K[0,0], K[0,1], K[0,2], K[1,1], K[1,2]
    
    # The params list will hold the 5 parameters of K and all the 6 parameters for all
    # the R's and t's (together) for all the patterns. 
    # So, length of params is 5 + 3*n + 3*n (n = number of patterns).
    params = [ alphax, s, x0, alphay, y0 ]

    # rectifyRadialDistortion is a flag that indicates if the radial distortion 
    # has to be rectified or not. This is true if some valid values of k1 and k2 
    # are provided.
    
    if k1 is not None and k2 is not None:   rectifyRadialDistortion = True
    else:       rectifyRadialDistortion = False
        
    if rectifyRadialDistortion:     params += [ k1, k2 ]    # Include k1 and k2.
    
    for i in range( len( setOfXij ) ):
        # Appending the elements of R and t in params
        R, t = setOfR[i], setOfT[i]
        w = rodriguesRtoW( R )
        params += [ w[0], w[1], w[2], t[0], t[1], t[2] ]
    
    params = np.array( params )

#-------------------------------------------------------------------------------

    def errorFunction( params ):
        '''
        This function takes in the parameters and using XWij, finds the 
        estimated values of Xij (XijEst). It then compares them with the actual Xij 
        values (which are the location of the corners of the pattern) and outputs 
        an error.
        '''
        alphax_1, s_1, x0_1, alphay_1, y0_1 = \
                        params[0], params[1], params[2], params[3], params[4]
        K_1 = np.array( [ [ alphax_1, s_1, x0_1 ], [ 0, alphay_1, y0_1 ], [ 0, 0, 1 ] ] )
        
        if rectifyRadialDistortion:
            iStart = 7
            k1_1, k2_1 = params[5], params[6]
        else:
            iStart = 5
        
        for i in range( iStart, len( params ), 6 ):
            R_1 = rodriguesWtoR( [ params[i], params[i+1], params[i+2] ] )
            t_1 = np.array( [ [ params[i+3] ], [ params[i+4] ], [ params[i+5] ] ] )
        
            P_1 = np.matmul( K_1, np.hstack( ( R_1, t_1 ) ) )
            
            # Suppose number of corners is 80.
            XWij_1 = np.array( XWij ).T     # Shape 80x4 transposed to 4x80.
            XijEstH = np.matmul( P_1, XWij_1 ).T    # Shape 3x80 transposed to 80x3.
        
            # Convert to planar.
            for j in range( XijEstH.shape[0] ):     XijEstH[j] /= XijEstH[j][2]
        
            XijEst = XijEstH[ :, :-1 ]    # Shape now 80x2. Last column removed. Planar form.
            
            # Index is reconfigured for list setOfXij.
            Xij_1 = setOfXij[ int( ( i - iStart ) / 6 ) ]

            Xij_1 = np.array( Xij_1 )    # Shape 80x2.
            
            #print(Xij_1.shape, XijEst.shape)
            
            # If rectifyRadialDistortion flag is True, then optimize the radial 
            # distortion parameters as well. So incorporate them in the error function.
            if rectifyRadialDistortion:
                x, y = XijEst[:, 0], XijEst[:, 1]
                rSq = (x - x0_1)**2 + (y - y0_1)**2
                xRad = x + (x - x0_1) * ( k1_1*rSq + k2_1*rSq**2 )
                yRad = y + (y - y0_1) * ( k1_1*rSq + k2_1*rSq**2 )
                
                # Remember that the xRad and yRad are all 80x1 vectors.
                XijEst[:, 0] = xRad
                XijEst[:, 1] = yRad
            
            error = Xij_1 - XijEst
            error = np.linalg.norm( error, axis=1 )     # Taking norm of error. Shape 80x1.
            
            # This will hold all the errors for all the 80 corners of 40 images. 
            # Total 3200 elements.
            errorVector = error if i == iStart else np.hstack( ( errorVector, error ) )
            
            # LM needs the number of residuals (i.e. the size of error vector) to be 
            # more or equal to the size of the variables (params) which are optimized.
            # Hence the error is reshaped the size of the params list (as in this case
            # for the 80 corners, we had error as 80x1 but total number of parameters
            # for 40 images was 5 (intrinsic) + 40 * 6 (extrinsic) = 245, or 
            # 7 (intrinsic) + 40 * 6 (extrinsic) = 247 (if k1 and k2 of radial 
            # distortion is considered)).

        #print(errorVector.shape)
        
        return errorVector

#-------------------------------------------------------------------------------

    # Optimized output.
    startTime = time.time()
    paramsNew = optimize.least_squares( errorFunction, params, method='lm' )
    print( f'Time taken by LM optimizer: {time.time() - startTime} sec.' )
    
    # The optimized vector is obtained as paramsNew.x.
    paramsNew = paramsNew.x
    #print( paramsNew )

    alphaxNew, sNew, x0new, alphayNew, y0new = \
                    paramsNew[0], paramsNew[1], paramsNew[2], paramsNew[3], paramsNew[4]
    Knew = np.array( [ [ alphaxNew, sNew, x0new ], [ 0, alphayNew, y0new ], [ 0, 0, 1 ] ] )
    
    if rectifyRadialDistortion:
        iStart = 7
        k1new, k2new = paramsNew[5], paramsNew[6]
    else:
        iStart = 5
        k1new, k2new = k1, k2

    setOfRnew, setOfTnew = [], []
    for i in range( iStart, len( paramsNew ), 6 ):
        Rnew = rodriguesWtoR( [ paramsNew[i], paramsNew[i+1], paramsNew[i+2] ] )
        tNew = np.array( [ [ paramsNew[i+3] ], [ paramsNew[i+4] ], [ paramsNew[i+5] ] ] )
        
        setOfRnew.append( Rnew )
        setOfTnew.append( tNew )
    
    return Knew, setOfRnew, setOfTnew, k1new, k2new
        
#===============================================================================

if __name__ == '__main__':
    
    # TASK: 2.2.1. Finding the corners.
    
    # Loading the images of given dataset.

    filepath = './Files/Dataset1'
    outputFilepath = './output_images'

    #filepath = './own_dataset'
    #outputFilepath = './output_image_own_dataset'
    
    listOfFiles = os.listdir( filepath )
    listOfFiles.sort()
    nPatterns = len( listOfFiles )
    
    # This is the V matrix for calculating the omega matrix.
    # And the homography (H) for every image is also stored in a separate list
    # as those will be used to calculate the R and t for every positon of the 
    # calibration pattern.
    # The intersection point lists for all the different pattern images are also
    # stored in a separate list.
    V, setOfH, setOfIntSctPtList = [], [], []
    
    for idx, i in enumerate( listOfFiles ):
        
        print( i )

        img = cv2.imread( os.path.join( filepath, i ) )    # Read images
        imgH, imgW, _ = img.shape
        
        gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )      # Convert to grayscale.
        
        blur = cv2.GaussianBlur( gray, (5,5), 0 )     # Blurring the image.
        
        # Find otsu's threshold value with.
        threshVal, otsu = cv2.threshold( blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
        
        #print( f'Threshold for color filter: {threshVal}' )
        
        # Making the white regions dilated and eroded, so that the unwanted 
        # black zones in the otsu image disappear.
        structElem = cv2.getStructuringElement( cv2.MORPH_RECT, ksize=( 5, 5 ) )
        dilated = cv2.dilate( otsu, kernel=structElem, anchor=(-1,-1) )
        eroded = cv2.erode( dilated, kernel=structElem, anchor=(-1,-1) )
        
        # Finding the edges.
        edgeThresh = 10
        edges = cv2.Canny( eroded, threshold1=edgeThresh, threshold2=3*edgeThresh, \
                                                            apertureSize=3 )
        
        ## Saving edge detected image.
        #cv2.imwrite( os.path.join( outputFilepath, i[:-4] + '_edges' + i[-4:] ), edges )
        
        # Finding the hough lines.
        # maxLineGap parameter seems to be the most important to merge redundant lines.
        lines = cv2.HoughLinesP( edges, rho=1, theta=math.pi/180, threshold=30, \
                                                    minLineLength=15, maxLineGap=250 )
        
        output = copy.deepcopy( img )
        
        nLines = lines.shape[0]     # No. of detected lines.
        #print( lines.shape )
        
#-------------------------------------------------------------------------------
        
        # Lists to save the vertical lines and horizontal lines and points.
        horiLinesPts, vertLinesPts, horiLinesH, vertLinesH = [], [], [], []
        
        # Draw the lines on the image.
        for l in lines:
            # Extending the x, y values of the end-points of the lines.
            x1, y1, x2, y2 = l[0,0], l[0,1], l[0,2], l[0,3]

            # Points in homogeneous coordinates.
            X1H, X2H = np.array( [x1,y1,1] ), np.array( [x2,y2,1] )
            LH = np.cross( X1H, X2H )   # Line in homogeneous coordinates.

            # Four corners of the image in homogeneous coordinates.
            tlH, trH = np.array( [0,0,1] ), np.array( [imgW-1,0,1] )
            blH, brH = np.array( [0,imgH-1,1] ), np.array( [imgW-1,imgH-1,1] )
            
            # Four boundaries of the image in homogeneous coordinates.
            topBoundaryH = np.cross( tlH, trH )
            botBoundaryH = np.cross( blH, brH )
            leftBoundaryH = np.cross( tlH, blH )
            rightBoundaryH = np.cross( trH, brH )
            #print( topBoundaryH, botBoundaryH, leftBoundaryH, rightBoundaryH )
            
            slopeAngle = math.atan2( y2-y1, x2-x1 )
            slopeAngleDeg = slopeAngle * 180 / math.pi
            
#-------------------------------------------------------------------------------

            # We have to extend the line segments to the boundary of the image.
            # So to find the points that they will intersect on the boundary when
            # extended we do the following.
            if abs( slopeAngleDeg ) < 45:
                # If the slope is less than 45 deg, the line will intersect better 
                # with the left and right boundary. 
                                
                # Finding the intersection point of the line with the left and right
                # boundary in homogeneous coordinates.
                
                intsctPtOfLwithLeftBound = np.cross( LH, leftBoundaryH )
                # Now converting the intersection point to the planar coordinates.
                x1new = intsctPtOfLwithLeftBound[0] / intsctPtOfLwithLeftBound[2]
                y1new = intsctPtOfLwithLeftBound[1] / intsctPtOfLwithLeftBound[2]
                
                #print(intsctPtOfLwithLeftBound, x1new, y1new, slopeAngle, x1, y1, x2, y2)
                
                intsctPtOfLwithRightBound = np.cross( LH, rightBoundaryH )                
                # Now converting the intersection point to the planar coordinates.
                x2new = intsctPtOfLwithRightBound[0] / intsctPtOfLwithRightBound[2]
                y2new = intsctPtOfLwithRightBound[1] / intsctPtOfLwithRightBound[2]
                
                x1new, y1new, x2new, y2new = int(x1new), int(y1new), int(x2new), int(y2new)
                #cv2.line( output, (x1new, y1new), (x2new, y2new), (0,255,255), 1 )
                
                horiLinesPts.append( [x1new, y1new, x2new, y2new] )
                horiLinesH.append( LH.tolist() )
                
#-------------------------------------------------------------------------------

            else:
                # And if the slope is more than 45 deg, the line may intersect 
                # better with the top and bottom boundary.

                # Finding the intersection point of the line with the top and bot
                # boundary in homogeneous coordinates.
                
                intsctPtOfLwithTopBound = np.cross( LH, topBoundaryH )
                # Now converting the intersection point to the planar coordinates.
                x1new = intsctPtOfLwithTopBound[0] / intsctPtOfLwithTopBound[2]
                y1new = intsctPtOfLwithTopBound[1] / intsctPtOfLwithTopBound[2]
                
                intsctPtOfLwithBotBound = np.cross( LH, botBoundaryH )
                # Now converting the intersection point to the planar coordinates.
                x2new = intsctPtOfLwithBotBound[0] / intsctPtOfLwithBotBound[2]
                y2new = intsctPtOfLwithBotBound[1] / intsctPtOfLwithBotBound[2]
                
                x1new, y1new, x2new, y2new = int(x1new), int(y1new), int(x2new), int(y2new)
                #cv2.line( output, (x1new, y1new), (x2new, y2new), (255,255,0), 1 )
                
                vertLinesPts.append( [x1new, y1new, x2new, y2new] )
                vertLinesH.append( LH.tolist() )
                
#-------------------------------------------------------------------------------

        # Sort the horizontal lines according to the y intercepts.
        horiLinesPtsSorted = sorted( horiLinesPts, key=lambda x: x[1] )
        # Also sorting the list of lines in homogeneous coordinates in the same manner.
        horiLinesHsorted = sorted( horiLinesH, key=lambda x: horiLinesPts[ horiLinesH.index(x) ][1] )
        
        horiLinesPts = horiLinesPtsSorted
        horiLinesH = horiLinesHsorted
        
        nHoriLines = len( horiLinesPts )
        #print( horiLinesPts )
        horiLinesPtsArr = np.array( horiLinesPts )
        
        # This array will have the values of the gap between the horizontal lines.
        horiLinesGaps = horiLinesPtsArr[ 1 : ] - horiLinesPtsArr[ 0 : nHoriLines-1 ]
        avgHoriLinesGap = np.mean( horiLinesGaps, axis=0 )

        # This array contains a record of which pair of consecutive lines has a 
        # gap less than half of the average gap. 
        # If line 2 and 3 has low gap, then the record 2 will be all false.
        # If lines 2, 3 and 4 has low gap between them, then record 2 and 3 will be 
        # all false.
        validHoriGaps = horiLinesGaps > avgHoriLinesGap * 0.5

        #print( validHoriGaps )
        
        # Will hold horiLinesPts after removing redundant lines.
        # The loop runs till nHoriLines-1 as there are 1 less gaps than the number 
        # of lines.
        pureHoriLinesPts = [ horiLinesPts[j] for j in range( nHoriLines-1 ) \
                                                if validHoriGaps[j][1] and validHoriGaps[j][3] ]
        pureHoriLinesH = [ horiLinesH[j] for j in range( nHoriLines-1 ) \
                                                if validHoriGaps[j][1] and validHoriGaps[j][3] ]
        # So long as the records corresponding to both the y intercepts are not True,
        # (both y intercepts meaning element 1 and 3 in the list) 
        # (which implies that the gap of the corresponding line with the next was
        # smaller than the average gap), igonre it and don't include it to the 
        # list of pureHoriLinesPts.

        # Include the last line which was not considered in the loop.
        pureHoriLinesPts.append( horiLinesPts[ nHoriLines - 1 ] )
        pureHoriLinesH.append( horiLinesH[ nHoriLines - 1 ] )
        
        print( f'Number of horizontal lines: {len(pureHoriLinesPts)}' )
        
        for j in pureHoriLinesPts:
            x1, y1, x2, y2 = j[0], j[1], j[2], j[3]
            cv2.line( output, (x1, y1), (x2, y2), (0,0,255), 2 )

#-------------------------------------------------------------------------------

        # Sort the vertzontal lines according to the y intercepts.
        vertLinesPtsSorted = sorted( vertLinesPts, key=lambda x: x[0] )
        # Also sorting the list of lines in homogeneous coordinates in the same manner.
        vertLinesHsorted = sorted( vertLinesH, key=lambda x: vertLinesPts[ vertLinesH.index(x) ][0] )
        
        vertLinesPts = vertLinesPtsSorted
        vertLinesH = vertLinesHsorted
        
        nVertLines = len( vertLinesPts )
        #print( vertLinesPts )
        vertLinesPtsArr = np.array( vertLinesPts )

        # This array will have the values of the gap between the vertzontal lines.
        vertLinesGaps = vertLinesPtsArr[ 1 : ] - vertLinesPtsArr[ 0 : nVertLines-1 ]
        avgVertLinesGap = np.mean( vertLinesGaps, axis=0 )

        # This array contains a record of which pair of consecutive lines has a 
        # gap less than half of the average gap. 
        # If line 2 and 3 has low gap, then the record 2 will be all false.
        # If lines 2, 3 and 4 has low gap between them, then record 2 and 3 will be 
        # all false.
        validVertGaps = vertLinesGaps > avgVertLinesGap * 0.5

        #print( validVertGaps )
        
        # Will hold vertLinesPts after removing redundant lines.
        # The loop runs till nVertLines-1 as there are 1 less gaps than the number 
        # of lines.
        pureVertLinesPts = [ vertLinesPts[j] for j in range( nVertLines-1 ) \
                                                if validVertGaps[j][0] and validVertGaps[j][2] ]
        pureVertLinesH = [ vertLinesH[j] for j in range( nVertLines-1 ) \
                                                if validVertGaps[j][0] and validVertGaps[j][2] ]
        # So long as the records corresponding to both the y intercepts are not True,
        # (both y intercepts meaning element 0 and 2 in the list) 
        # (which implies that the gap of the corresponding line with the next was
        # smaller than the average gap), igonre it and don't include it to the 
        # list of pureVertLinesPts.

        # Include the last line which was not considered in the loop.
        pureVertLinesPts.append( vertLinesPts[ nVertLines - 1 ] )
        pureVertLinesH.append( vertLinesH[ nVertLines - 1 ] )
        
        print( f'Number of vertical lines: {len(pureVertLinesPts)}' )
        
        for j in pureVertLinesPts:
            x1, y1, x2, y2 = j[0], j[1], j[2], j[3]
            cv2.line( output, (x1, y1), (x2, y2), (255,0,0), 2 )

        ## Saving line detected image.
        #cv2.imwrite( os.path.join( outputFilepath, i[:-4] + '_lines' + i[-4:] ), output )

#-------------------------------------------------------------------------------

        # Now finding the intersection points.
        corners = copy.deepcopy( img )
        intsctPtIdx, intsctPtList = 0, []       # List and index of the intersection points.
        for hl in pureHoriLinesH:
            for vl in pureVertLinesH:
                intsctPtIdx += 1
                ptH = np.cross( hl, vl )    # Intersection point in homogeneous coordinates.
                pt = [ int( ptH[0] / ptH[2] ), int( ptH[1] / ptH[2] ) ]     # Planar form.
                intsctPtList.append( pt )
                cv2.circle( corners, tuple(pt), 3, (0,255,0), -1 )      # Draw corners.
                cv2.putText( corners, str(intsctPtIdx), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, \
                                                                    0.35, (255, 255, 0), 1 )
        
        setOfIntSctPtList.append( intsctPtList )    # Storing the intersection point lists.
        
        nIntsctPt = len( intsctPtList )
        print( f'Number of intersection points: {nIntsctPt}' )
        
        ## Saving corner detected image.
        #cv2.imwrite( os.path.join( outputFilepath, i[:-4] + '_corners' + i[-4:] ), corners )

#-------------------------------------------------------------------------------

        #cv2.imshow( 'img', img )
        #cv2.imshow( 'blur', otsu )
        #cv2.imshow( 'eroded', eroded )
        #cv2.imshow( 'edges', edges )
        #cv2.imshow( 'output', output )
        #cv2.imshow( 'corners', corners )
        #key = cv2.waitKey(0)
        #if key & 0xFF == 27:    break   # esc to break.

#-------------------------------------------------------------------------------

        # TASK: 2.2.2. Calibration.
        
        # World coordinate values for the corner points are also calculated.
        # This will stay the same for all the patterns and hence it is only 
        # calculated once (during the processing of the first pattern image)
        # and then skipped in the other iterations.
        if idx == 0:
            
            worldPtList = []
            
            x00, y00 = 0, 0       # The physical measurement in mm for the 1st corner.
            
            # These are the lengths of the sides of the black squares.
            # So the corners are separated by this distance in the world coordinate.
            xStep, yStep = 25, 25
            
            nStepsAlongX = len( pureVertLinesPts )      # No. of squares along x.
            nStepsAlongY = len( pureHoriLinesPts )      # No. of squares along y.
            
            for pIdx, pt in enumerate( intsctPtList ):
                xW = x00 + xStep * ( pIdx % nStepsAlongX )
                yW = y00 + yStep * int( pIdx / nStepsAlongX )
                worldPtList.append( [ xW, yW ] )
                #print( xW, yW )
                
#-------------------------------------------------------------------------------

        ## Convert to homogeneous coordinate format.
        #intsctPtListH = [ [ pt[0], pt[1], 1 ] for pt in intsctPtList ]
        
        # Xi = H * Xw equation (xi and xw known) is written as Ah = 0, to solve for h.
        # h is the vector [h11, h12, h13, h21, h22, h23, h31, h32, h33].
        A = []
        for i in range( nIntsctPt ):
            xw, yw = worldPtList[ i ][0], worldPtList[ i ][1]
            xi, yi = intsctPtList[ i ][0], intsctPtList[ i ][1]
            
            A.append( [ xw, yw, 1, 0, 0, 0, -xi*xw, -xi*yw, -xi ] )
            A.append( [ 0, 0, 0, xw, yw, 1, -yi*xw, -yi*yw, -yi ] )
            
        # Finding the homography for the current image.
        A = np.array( A )
        
        # Ah = 0 is a homogeneous equation, so a least square minimization of this
        # overdetermined system, without any constraint will always give the trivial
        # h=0 solution. To prevent it a contraint of ||h|| = 1 is applied. 
        # Now, the norm of h could have been something else also, but it will always 
        # be a scalar number. Hence without loss of generality we can assume this to 
        # be 1. This is because any other norm value will just be a scalar multiple of
        # this and also no matter what multiple it is, it will not affect the equation
        # Ah = 0. As, the equation stays the same even after multiplying a scalar to h.
        
        # To find the h, svd is done on A and the last column of Vm matrix will give
        # the non-trivial h.
        Um, D, VmT = np.linalg.svd( A )
        
        h = np.transpose( VmT )[:,-1]
        H = np.reshape( h, (3,3) )
        H = H / H[2,2]
        
        setOfH.append( H )      # Storing the H in a list.

#-------------------------------------------------------------------------------

        # Finding the rows of the V and b matrix for the current image.
        
        V12 = np.array( [ H[0,0]*H[0,1], \
                          H[0,0]*H[1,1] + H[1,0]*H[0,1], \
                          H[1,0]*H[1,1], \
                          H[2,0]*H[0,1] + H[0,0]*H[2,1], \
                          H[2,0]*H[1,1] + H[1,0]*H[2,1], \
                          H[2,0]*H[2,1] ] )
    
        V11 = np.array( [ H[0,0]*H[0,0], \
                          H[0,0]*H[1,0] + H[1,0]*H[0,0], \
                          H[1,0]*H[1,0], \
                          H[2,0]*H[0,0] + H[0,0]*H[2,0], \
                          H[2,0]*H[1,0] + H[1,0]*H[2,0], \
                          H[2,0]*H[2,0] ] )
    
        V22 = np.array( [ H[0,1]*H[0,1], \
                          H[0,1]*H[1,1] + H[1,1]*H[0,1], \
                          H[1,1]*H[1,1], \
                          H[2,1]*H[0,1] + H[0,1]*H[2,1], \
                          H[2,1]*H[1,1] + H[1,1]*H[2,1], \
                          H[2,1]*H[2,1] ] )

        V.append( V12.tolist() )
        V.append( ( V11 - V22 ).tolist() )

#-------------------------------------------------------------------------------

    V = np.array(V)
    #print( V.shape )
    
    # Now, finding the omega. V*omega = 0 is the equation, which is again an 
    # overdetermined homogeneous system of equation, the non-trivial solution of 
    # which can be found by imposing a constraint of ||omega|| = 1, (without loss
    # of generality) using the svd of V and then taking the last column of the Vm
    # matrix.
    Um, D, VmT = np.linalg.svd( V )
    
    omega = np.transpose( VmT )[:,-1]
    Omega = np.array( [ [ omega[0], omega[1], omega[3] ], \
                        [ omega[1], omega[2], omega[4] ], \
                        [ omega[3], omega[4], omega[5] ] ] )

    #print( f'Omega:\n{Omega}' )
    
#-------------------------------------------------------------------------------

    # Finding the value of K intrinsic parameter matrix by Zhang's algorithm.
    y0num = Omega[0,1] * Omega[0,2] - Omega[0,0] * Omega[1,2]
    y0den = Omega[0,0] * Omega[1,1] - Omega[0,1]**2
    y0 = y0num / y0den
    Lambda = Omega[2,2] - ( Omega[0,2]**2 + y0 * y0num ) / Omega[0,0]
    alphax = math.sqrt( Lambda / Omega[0,0] )
    alphay = math.sqrt( Lambda * Omega[0,0] / y0den )
    s = -1 * Omega[0,1] * (alphax**2) * alphay / Lambda
    x0 = s * y0 / alphay - Omega[0,2] * (alphax**2) / Lambda
    
    K = np.array( [ [ alphax, s, x0 ], [ 0, alphay, y0 ], [ 0, 0, 1 ] ] )
    
    #print( f'intrinsic parameter matrix K:\n{K}' )
    
#-------------------------------------------------------------------------------
        
    # Finding the R matrix and the t vector for every pattern.
    Kinv = np.linalg.inv( K )
    
    # The rotation matrix (R) and the translation vector (t) for every image is 
    # also stored in a separate list.
    setOfR, setOfT = [], []
    
    for idx, i in enumerate( listOfFiles ):
        H = setOfH[ idx ]     # Homography matrix for ith pattern.
        
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]        
        r1 = np.matmul( Kinv, h1 )
        
        r2 = np.matmul( Kinv, h2 )
        r3 = np.cross( r1, r2 )
        t = np.matmul( Kinv, h3 )
        
        eta1 = 1 / np.linalg.norm( r1 )      # Reciprocal of the norm of r1.
        eta2 = 1 / np.linalg.norm( r2 )      # Reciprocal of the norm of r1.
        
        eta = ( eta1 + eta2 ) * 0.5
        
        # Ideally the eta1 should be equal to eta2, but they will not be the same
        # due to noise in the measurement. So, either eta1 or eta2 can be taken 
        # and the final LM optimization will take care of the error and fix it.
        # But we found that when eta = eta1 + eta2, the overall error was better 
        # compared to when eta = eta1 or eta = eta2. Although it took more time
        # for the LM optimization to converge with this.
        
        # This will be used to scale the R and t elements.        
        r1, r2, r3, t = eta * r1, eta * r2, eta * r3, eta * t
        
        # Reshaping r's and t into 3x1 vectors from arrays of size (3,).
        r1 = np.reshape( r1, (3,1) )
        r2 = np.reshape( r2, (3,1) )
        r3 = np.reshape( r3, (3,1) )
        t = np.reshape( t, (3,1) )
        
        R = np.hstack( (r1, r2, r3) )

        # Now conditioning R so that it is orthonormal by using svd.
        Um, D, VmT = np.linalg.svd( R )
        R = np.matmul( Um, VmT )    # This is the orthonormal version of R.
        
        setOfT.append( t )      # Storing the t in a list.
        setOfR.append( R )      # Storing the R in a list.
        
        #print( f'{i}, R:\n{R}' )
        
#-------------------------------------------------------------------------------

    # Radial distortion parameters are initilized to 0.
    k1, k2 = 0, 0

#-------------------------------------------------------------------------------

    # Using Levenberg-Marqdth algorithm for finding the refined values of 
    # R, K, t. For this we have to send their elements as a vector into the 
    # error function.
    setOfXij = setOfIntSctPtList
    XWij = worldPtList
    
    Knew, setOfRnew, setOfTnew, k1new, k2new = \
                        refineParamByLM( setOfXij, XWij, K, setOfR, setOfT, k1, k2 )
                    
    print( f'K:\n{K}' )
    print( f'Refined K:\n{Knew}' )
    print( f'k1: {k1}, k2: {k2}' )
    print( f'Refined k1: {k1new}, Refined k2: {k2new}' )
                    
    #print( len(setOfR), len(setOfRnew), len(setOfT), len(setOfTnew) )    

#-------------------------------------------------------------------------------

    # Reprojction of the points to the image to check the amount of refinement.
    # The mean and variance of the error in position is also calculated (both
    # with and without optimization).

    # Convert worldPtList to homogeneous format.
    XWij = [ [ pt[0], pt[1], 0, 1 ] for pt in worldPtList ]
    # zw is 0 for all the corners as the pattern is assumed to be in z=0 plane.

    XWij = np.array( XWij ).T     # Shape 80x4 transposed to 4x80.

    totalError, totalSqError, totalErrorLm, totalSqErrorLm = 0, 0, 0, 0
    
    for idx, i in enumerate( listOfFiles ):
        # First estimating the position based on the unoptimized parameters.
        img = cv2.imread( os.path.join( filepath, i ) )    # Read images
        
#-------------------------------------------------------------------------------

        R, t = setOfR[idx], setOfT[idx]
        P = np.matmul( K, np.hstack( ( R, t ) ) )
        
        XijEstH = np.matmul( P, XWij ).T    # Shape 3x80 transposed to 80x3.
        
        # Convert to planar.
        for j in range( XijEstH.shape[0] ):     XijEstH[j] /= XijEstH[j][2]
    
        # Suppose number of corners is 80.
        XijEst = XijEstH[ :, :-1 ]    # Shape now 80x2. Last column removed. Planar form.
        Xij = setOfIntSctPtList[ idx ]
        Xij = np.array( Xij )    # Shape 80x2.
        
        # Incorporate radial distortion parameters as well.
        x, y, x0, y0 = XijEst[0], XijEst[1], K[0,2], K[1,2]
        rSq = (x - x0)**2 + (y - y0)**2
        xRad = x + (x - x0) * ( k1 * rSq + k2 * rSq**2 )
        yRad = y + (y - y0) * ( k1 * rSq + k2 * rSq**2 )
        
        # Remember that the xRad and yRad are all 80x1 vectors.
        XijEst[0] = xRad
        XijEst[1] = yRad
        
        error = Xij - XijEst        # This is also an 80x1 vector.
        error = np.linalg.norm( error, axis=1 )     # Taking norm of error. Shape 80x1.
        errorSq = error**2
        
        error = np.mean( error )    # This error is now the euclidean distance between 
        # the actual and estimated Xij value.
        # The mean is calculated and not the sum, as this is an 80x1 vector.
        variance = np.mean( errorSq ) - error**2
        
        totalSqError += error**2
        totalError += error

#-------------------------------------------------------------------------------

        Rlm, tLm = setOfRnew[idx], setOfTnew[idx]
        Plm = np.matmul( Knew, np.hstack( ( Rlm, tLm ) ) )
        
        XijEstHlm = np.matmul( Plm, XWij ).T    # Shape 3x80 transposed to 80x3.
        
        # Convert to planar.
        for j in range( XijEstHlm.shape[0] ):     XijEstHlm[j] /= XijEstHlm[j][2]
    
        # Suppose number of corners is 80.
        XijEstLm = XijEstHlm[ :, :-1 ]    # Shape now 80x2. Last column removed. Planar form.
        
        # Incorporate radial distortion parameters as well.
        x, y, x0, y0 = XijEstLm[0], XijEstLm[1], Knew[0,2], Knew[1,2]
        rSq = (x - x0)**2 + (y - y0)**2
        xRad = x + (x - x0) * ( k1new * rSq + k2new * rSq**2 )
        yRad = y + (y - y0) * ( k1new * rSq + k2new * rSq**2 )
        
        # Remember that the xRad and yRad are all 80x1 vectors.
        XijEstLm[0] = xRad
        XijEstLm[1] = yRad
        
        errorLm = Xij - XijEstLm        # This is also an 80x1 vector.
        errorLm = np.linalg.norm( errorLm, axis=1 )     # Taking norm of error. Shape 80x1.
        errorSqLm = errorLm**2
        
        errorLm = np.mean( errorLm )    # This error is now the euclidean distance between 
        # the actual and estimated Xij value after optimization.
        # The mean is calculated and not the sum, as this is an 80x1 vector.
        varianceLm = np.mean( errorSqLm ) - errorLm**2
        
        totalSqErrorLm += errorLm**2
        totalErrorLm += errorLm

#-------------------------------------------------------------------------------

        # Plot the points.
        mapped1 = copy.deepcopy( img )   # Copy of the image on which points will be drawn.
        mapped2 = copy.deepcopy( img )   # Copy of the image on which points will be drawn.

        for pIdx, pt in enumerate( setOfIntSctPtList[ idx ] ):
            cv2.circle( mapped1, tuple(pt), 3, (0,0,255), -1 )
            #print( int(XijEst[pIdx][0]), int(XijEst[pIdx][1]) )
            cv2.circle( mapped1, ( int(XijEst[pIdx][0]), int(XijEst[pIdx][1]) ), 4, (255,0,0), 2 )
            cv2.circle( mapped2, tuple(pt), 3, (0,0,255), -1 )
            #print( int(XijEstLm[pIdx][0]), int(XijEstLm[pIdx][1]) )
            cv2.circle( mapped2, ( int(XijEstLm[pIdx][0]), int(XijEstLm[pIdx][1]) ), 4, (0,255,0), 2 )
        
        mapped = np.hstack( ( mapped1, mapped2 ) )
        cv2.imshow( 'mapped (left: without LM; right: after LM)', mapped )
        key = cv2.waitKey(50)
        cv2.destroyAllWindows()

        # Saving edge detected image.
        cv2.imwrite( os.path.join( outputFilepath, i[:-4] + '_mapped' + i[-4:] ), mapped )
        
#-------------------------------------------------------------------------------

        print( f'Image: {i}' )
        print( f'Rotation Matrix (without LM):\n{R}' )
        print( f'Translation Vector (without LM):\n{t}' )
        print( f'Reprojection error (without LM) Mean: {error}, Variance: {variance}' )
        print( f'Rotation Matrix (after LM optimization):\n{Rlm}' )
        print( f'Translation Vector (after LM optimization):\n{tLm}' )
        print( f'Reprojection error (after LM optimization) Mean: {errorLm}, Variance: {varianceLm}' )

#-------------------------------------------------------------------------------

    meanError = totalError / nPatterns
    errorVariance = totalSqError / nPatterns - meanError**2
    
    print( f'Mean Error: {meanError} (without optimization)' )
    print( f'Error Variance: {errorVariance} (without optimization)' )
        
    meanErrorLm = totalErrorLm / nPatterns
    errorVarianceLm = totalSqErrorLm / nPatterns - meanErrorLm**2
    
    print( f'Mean Error: {meanErrorLm} (after LM optimization)' )
    print( f'Error Variance: {errorVarianceLm} (after LM optimization)' )
        




