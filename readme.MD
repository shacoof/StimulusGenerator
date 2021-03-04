use l to switch control to virtual-screen location , arrows keys will change VS locaiton
use s to switch control to virtual-screen size , arrows keys will change VS size 
supported keyboard actions 
    ? = for help 
    l - switch to virtual screen location setting
    s - switch to virtual screen size setting 
    <Up> - move up or reduce virtual screen height
    <Down> - move down or extend virtual screen hegith
    <Left>  - move left or reduce virtual screen width
    <Right> - move right or extend virtual screen size 
    u - to update app config with current virtual screen location

    r - run the simulation, pressing r again will restart the simulaiton from the begining and reload the latest version of the CSV
    p - pause the simulation 

    e - to exit


appConfig.csv
fishScreenStartX
fishScreenStartY
fishScreenWidth
fishScreenHeight
f9CommunicationEnabled - if on then we try to communicate with a server running on the Stream-7 machine (needs to be up and running). Note that if the server is not running then the program will hangs as the communication is synchronous 
DishRadiusSize = the radius of the fish-dish (mm), used for calculaiton of degress / object size / speed 
VirtualScreenDegrees = how many (out of 360 degress) the virtual screen is taking. speed is measured in degrees per second. 
VirtualScreenWidthActualSize = the size in mm of the VirtualScreen,  it is circa 2*pi*DishRadiusSize/2 but we don't caluclate it to allow flexability as the virtual screen captures less than 1/2 

stimulusConfig.csv
The coordinates of the virtual screen are between 0-999. top left corenr is (0,0)
We automatically translates these coordinates to the real size of the virtual screen, which might change depending on the setup
example : if the real virtual screensize is 100x50 then all coordinates between (0,0) and (99,49) will be mapped to (0,0)
the formula is trunc(x/virtual screen width) and trunc(y/virtual screen height)

Shape	: Circle 
startX,startY,endX,endY : start and end points of shape track
StartX : x=0 is the top-left corner of the fishScreen as defined in appConfig.csv fishScreen parameters 
StartY : y=0 is the top-left corner of the fishScreen as defined in appConfig.csv fishScreen parameters 
repetition : how many times the stimulus will run 
startShapeRadius,endShapeRadius : measured in degrees 
fastDuration,slowDuration : distribution of time (in ms) for each phase 
startMode : AFTER the previous stimulus completed or starting WITH the previous stimulus, in both cases delay (in ms) can be applied 
fastSpeed/slowSpeed = measured in degrees per second 
exitCriteria : if Distance then the stimulus ends when the shpae reaches the endX / endY. if Time then stimulus ends after X ms (as given by duration column)
euration is ignored for tiem based stimuli
if distance is used then duration is ignored 
if time is used then distance is also applicable, i.e. we don't allow the stimulus to go beyond endX,endY
change in size is determined based on the exitCriteria 
if time then duration determines the pace of the change
if distance then it will be based on the time it will take the sahpe to travel from start to end 


formula 
object size (shapeRadius) is measured in degrees. We translate degrees to mm as follows
    ObjectSize in mm =  2*DishRadiusSize*tan(shapeRadius/2) 
we translate mm to pixels as follows
    ObjectSize in pixels =  (vsWidth[in pixels] * objectSize[in mm]) / VirtualScreenWidthActualSize[in mm]




  ActionItem add reload of csv DONE
  ActionItem show/hide cross DONE
  ActionItem changing speed Done
  ActionItem subsequent stimuli are running with delay>=0 with previous one , like power-point DONE
  ActionItem increase size linearly 
  ActionItem add a label at the top with current stimulus info


  alternatives
  http://www.portugueslab.com/stytra/userguide/1_examples_gallery.html
  http://psychtoolbox.org/docs/MorphDemo