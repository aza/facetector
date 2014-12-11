function TrackedObject(rect, flowTracker){
  var lastUpdate = new Date(),
      createdAt = new Date(),
      confidences = [rect.confidence],
      self = this

  rect = _(rect).clone()

  this.id = Math.floor(Math.random()*0xffffff).toString(16)
  this.rect = rect

  this.center = function(){
    return {x:rect.x+rect.width/2, y:rect.y+rect.height/2}
  }
  
  this.isInside = function(obj){
    var center = self.center(),
        isContainedX = center.x >= obj.rect.x && center.x <= obj.rect.x+obj.rect.width,
        isContainedY = center.y >= obj.rect.y && center.y <= obj.rect.y+obj.rect.height

    if( isContainedX && isContainedY ) return true
    return false
  }

  this.update = function( newRect ){
    rect = _(newRect).clone()

    if( newRect.confidence > 0){
      confidences.push(newRect.confidence)
      if( confidences.length > 5 ) confidences.shift()
    
      this.rect = rect
      lastUpdate = new Date()

      if( self.flowPoint ){
        self.flowPoint.setPoint(self.center())
      }
    }

    // If there isn't an active flow point, make a new flow point
    if( self.flowPoint === undefined || self.flowPoint.isActive() == false ){
      // Only start flow tracking if the face has been around for a bit
      if( new Date()-createdAt > 1000 ){
        self.flowPoint = flowTracker.addPointToTrack( self.center() )
      }
    }

  }

  this.isActive = function(){
    return (new Date()-lastUpdate) <= 500 ? true : false
  }

  this.confidence = function(){
    return _(confidences).reduce(function(num,memo){return num+memo},0)/confidences.length
  }

}

function Tracker( flowTracker ){
  var faces = []

  function getContainedFace( obj ){
    // TODO: Should figure out best contained face
    for( var i=0; i<faces.length; i++){
      if( faces[i].isInside(obj) ) return faces[i]
      if( faces[i].flowPoint && faces[i].flowPoint.isInside(obj) ){
        return faces[i]
      }
    }

    return null
  }

  function removeInactiveFaces(){
    faces = _(faces).filter(function(face){ return face.isActive() || (face.flowPoint && face.flowPoint.isActive()) })
  }

  this.update = function( rect ){
    removeInactiveFaces()

    var obj = new TrackedObject(rect, flowTracker)
    var matchedFace = getContainedFace(obj)

    if( matchedFace == null ){
      if( rect.confidence > 0 ){
        faces.push( obj )
        obj.update( rect )
      } 
    }
    else {
      matchedFace.update( rect )
    }

  }


  this.draw = function(ctx, scale){
    
    _(faces).each(function(face){
      ctx.globalAlpha = .4
      ctx.fillStyle = "#"+face.id

      if( face.isActive() ){
        ctx.fillRect(face.rect.x*scale, face.rect.y*scale, face.rect.width*scale, face.rect.height*scale)
        ctx.fillStyle = "#fff"
        ctx.fillText(parseInt(face.confidence()), face.rect.x*scale, face.rect.y*scale)  
      }
      else if( face.flowPoint && face.flowPoint.isActive() ){
        var point = face.flowPoint.point()
        ctx.fillRect(point.x*scale-20, point.y*scale-20, 40, 40)
      }

      if( face.flowPoint && face.flowPoint.isActive() ){
        ctx.globalAlpha = 1
        ctx.fillStyle = "#fff"
        var point = face.flowPoint.point()
        ctx.beginPath();
        ctx.arc(point.x*scale, point.y*scale, 4, 0, Math.PI*2, true); 
        ctx.closePath();
        ctx.fill();
      }

    })

    ctx.globalAlpha = 1
  }

}



function FlowTracker(width, height, ctx){
  var curr_img_pyr = new jsfeat.pyramid_t(3),
      prev_img_pyr = new jsfeat.pyramid_t(3),
      maxPointsToTrack = 5,
      flow = this
  
  curr_img_pyr.allocate(width, height, jsfeat.U8_t|jsfeat.C1_t)
  prev_img_pyr.allocate(width, height, jsfeat.U8_t|jsfeat.C1_t)

  
  var point_count = 0,
      point_status = new Uint8Array(maxPointsToTrack),
      prev_xy = new Float32Array(maxPointsToTrack*2),
      curr_xy = new Float32Array(maxPointsToTrack*2)

  var opt = {
    win_size: 30,
    max_iters: 5,
    epsilon: .01,
    min_eigen: .001
  }

  function FlowPoint( index ){
    var self = this

    this.point = function(){
      return {
        x: curr_xy[index<<1],
        y: curr_xy[(index<<1)+1]
      }
    }

    this.isActive = function(){
      return point_status[index] === 1
    }

    this.isInside = function(obj){
      var point = self.point(),
      isContainedX = point.x >= obj.rect.x && point.x <= obj.rect.x+obj.rect.width,
      isContainedY = point.y >= obj.rect.y && point.y <= obj.rect.y+obj.rect.height

      if( isContainedX && isContainedY ) return true
      return false
    }

    this.setPoint = function(point){
      curr_xy[index<<1] = point.x
      curr_xy[(index<<1)+1] = point.y

      prev_xy[index<<1] = point.x
      prev_xy[(index<<1)+1] = point.y
    }

  }

  incrementPointCount = function(){
    point_count = (point_count+1)%maxPointsToTrack;
  }

  this.addPointToTrack = function( point ){

    for(var i=0; point_status[point_count] == 1 && i<=maxPointsToTrack; i++){
      incrementPointCount()
    }

    curr_xy[point_count<<1] = point.x;
    curr_xy[(point_count<<1)+1] = point.y;
    var flowPoint = new FlowPoint( point_count )
    // This will break if we wrap all the way around
    incrementPointCount()
    return flowPoint
  }

  function pruneNonActivePoints(){
    for( var i=0; i<point_status.length; i++){
      if( point_status[i] == 0 ){
        curr_xy[i<<1] = 0
        curr_xy[(i<<1)+1] = 0
      }
    }
  }

  this.update = function(){
    var imageData = ctx.getImageData(0, 0, width, height);

    // swap flow data
    var _pt_xy = prev_xy;
    prev_xy = curr_xy;
    curr_xy = _pt_xy;
    var _pyr = prev_img_pyr;
    prev_img_pyr = curr_img_pyr;
    curr_img_pyr = _pyr;

    //stat.start("grayscale");
    jsfeat.imgproc.grayscale(imageData.data, width, height, curr_img_pyr.data[0]);
    //stat.stop("grayscale");

    //stat.start("build image pyramid");
    curr_img_pyr.build(curr_img_pyr.data[0], true);
    //stat.stop("build image pyramid");

    //stat.start("optical flow lk");
    jsfeat.optical_flow_lk.track(prev_img_pyr, curr_img_pyr, prev_xy, curr_xy, point_count, opt.win_size, opt.max_iters, point_status, opt.epsilon, opt.min_eigen);
    //stat.stop("optical flow lk");

    pruneNonActivePoints()
    //console.log( curr_xy, point_status)
  }

}

function FaceFinder( videoId ){
  var video = document.getElementById(videoId),
      tracker = null,
      flow = null,
      self = this
  
  var findVideoSize = function() {
    video.removeEventListener('loadeddata', findVideoSize);
    demo_app(video.videoWidth, video.videoHeight);
    compatibility.requestAnimationFrame(tick);
  }

  video.addEventListener('loadeddata', findVideoSize);

  compatibility.getUserMedia({video: true}, function(stream) {
      video.src = compatibility.URL.createObjectURL(stream);
      video.play();
  }, function (error){});

  var stat = new profiler();

  var img_u8,work_canvas,work_ctx,previousPyr;

  var max_work_size = 320;

  function demo_app(videoWidth, videoHeight) {
    
    var scale = Math.min(max_work_size/videoWidth, max_work_size/videoHeight);
    var w = videoWidth*scale;
    var h = videoHeight*scale;

    img_u8 = new jsfeat.matrix_t(w, h, jsfeat.U8_t | jsfeat.C1_t);
    work_canvas = document.createElement('canvas');
    document.body.appendChild(work_canvas)
    work_canvas.width = w;
    work_canvas.height = h;

    work_ctx = work_canvas.getContext('2d');
    work_ctx.fillStyle = "rgba(0,255,0,.5)"

    flow = new FlowTracker( w, h, work_ctx )
    tracker = new Tracker( flow )

    jsfeat.bbf.prepare_cascade(jsfeat.bbf.face_cascade);

    stat.add("bbf detector");
  }

  function tick() {

    //setTimeout(function(){
      compatibility.requestAnimationFrame(tick);  
    //}, 10)
    
    stat.new_frame();
    if (video.readyState !== video.HAVE_ENOUGH_DATA) return

    work_ctx.drawImage(video, 0, 0, work_canvas.width, work_canvas.height);
    var imageData = work_ctx.getImageData(0, 0, work_canvas.width, work_canvas.height);

    stat.start("bbf detector");

    jsfeat.imgproc.grayscale(imageData.data, work_canvas.width, work_canvas.height, img_u8);
    var pyr = jsfeat.bbf.build_pyramid(img_u8, 24*1, 24*1, 1);
    var faceRects = jsfeat.bbf.detect(pyr, jsfeat.bbf.face_cascade);
    faceRects = jsfeat.bbf.group_rectangles(faceRects, 1);

    // Sort the faces on confidence
    jsfeat.math.qsort(faceRects, 0, faceRects.length-1, function(a,b){return (b.confidence<a.confidence)})

    _(faceRects).each(function(faceRect){
      tracker.update( faceRect )
    })

    tracker.draw(work_ctx, work_canvas.width/img_u8.cols)
    flow.update()

    stat.stop("bbf detector");

    document.getElementById('log').innerHTML = stat.log();
  }

}