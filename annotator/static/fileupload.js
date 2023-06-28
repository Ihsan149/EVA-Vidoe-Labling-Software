$(function () {
    var message;
    var filename;
    var files = [];
    var num_files = 0;
    
    $('#uploadVideoBtn').on('mousedown', function(e){
        e.stopPropagation();
            validate_name($('#video_name')).done(function(validated) {
            if (validated) {
                if (files.length == 0) {
                    $('#fileError').parent().removeClass('has-success').addClass('has-error');
                    $('#fileError').text('No files selected').removeClass('hidden');
                } else {
                    parent = $('#video_name').parent();
                    filename = $('#video_name').val();
                    $.post("/create_video/" + filename + "/",
                           {project: $('#projectSelect').val()})
                     .done(function(data) {
                         if (data['status'] == 'success') {
                            //disable all fields during upload
                            disable_modal_fields();
                            for (var i = 0; i < files.length; i++) {
                                 files[i].submit();
                             }
                         } else {
                            parent.removeClass("has-sucess").addClass("has-error");
                            $('#nameError').text('Name already used').removeClass('hidden');
                        }
                   });
                }
            }
        });
    })
    
    $('#createVideoModal').on('hidden.bs.modal', function() {
        $(this).find('.form-group').removeClass('has-error');
        $(this).find('.form-group').removeClass('has-success');
        $(this).find('.help-block').removeClass('hidden').addClass('hidden');
        $('#fileUploadProgress').addClass('hidden');
        $("#projectSelect")[0].selectedIndex = 0
        $('#video_name').val('');
        files = [];
    });

   function disable_modal_fields() {
           $('#projectSelect').prop("disabled", true);
           $('#fileupload').prop("disabled", true);
           $('#uploadVideoBtn').prop("disabled", true);
           $('#video_name').prop("disabled", true);
           $('#closeVideoBtn').prop("disabled", true);
    };


    function validate_name($textbox) {
        let validated = $.Deferred();
        let error = true;
        let name = $textbox.val();
        let parent = $textbox.parent();
        if (/^[A-Za-z_0-9\b]+$/.test(name)) {
            if (name.length > 100) {
                parent.removeClass("has-success").addClass("has-error");
                $('#nameError').text('Use less than 100 characters').removeClass('hidden');
            } else {
                $.get("/check_video_name/" + name + "/")
                 .done(function(resp){
                    if (resp.nameAvailable) {
                        parent.removeClass("has-error").addClass("has-success");
                        $('#nameError').addClass('hidden');
                        validated.resolve(true);
                    } else {
                        parent.removeClass("has-success").addClass("has-error");
                        $('#nameError').text('Name already used').removeClass('hidden');
                        validated.resolve(false);
                    }
                 })
                 .fail(function(){
                     parent.removeClass("has-success").addClass("has-error");
                     $('#nameError').text('Something went wrong').removeClass('hidden');
                     validated.resolve(false);
                 });
             }
        } else {
            parent.removeClass("has-success").addClass("has-error");
            $('#nameError').text('Only letters, numbers and underscore allowed').removeClass('hidden');
            validated.resolve(false);
        }
        return validated;
    }
    
    $("#video_name").on("blur", function(){
        validate_name($(this));
    });

  $("#fileupload").fileupload({
    dataType: 'json',
    autoUpload: false,
    sequentialUploads: false,
    singleFileUploads: false
  })
  .bind('fileuploadadd', function(e, data) {
    var errorText = '';
    var imageTypes = /(\.|\/)(gif|jpe?g|png)$/i;
    var videoTypes = /(\.|\/)(mp4|mov|quicktime|avi)$/i;

    $.each(data.files, function(idx, val) {
      let fileType = val['type'];
      if (!(imageTypes.test(fileType) || videoTypes.test(fileType))) {
        errorText = 'File type not accepted';
      } else if (videoTypes.test(fileType) && (data.originalFiles.length > 1 || files.length >= 1)){
        errorText = 'Only 1 video file accepted';
      }
      
      if (errorText) {
        $('#fileError').parent().removeClass('has-success').addClass('has-error');
        $('#fileError').text(errorText).removeClass('hidden');
        $('#fileCount').removeClass('hidden').addClass('hidden');
        data.abort();
        return false
      }
    })
    if (!errorText) {
        if (files.length >= 1)
            files.pop(); // on multiple click of add files, take the recent one.
        files.push(data);
      num_files = 0;
      $.each(files, function(idx, val) {
          num_files += val.files.length;
      })
      
      $('#fileCount').removeClass('hidden');
      $("#fileCount").text(num_files + " files selected.");
      $('#fileError').parent().removeClass('has-error').addClass('has-success');
      $('#fileError').addClass('hidden');
    }
  })
  .bind('fileuploadsend', function (e, data) {
        data.url = 'upload/' + $('#video_name').val() + '/';
        $('#fileUploadProgress').removeClass('hidden');
  })
  .bind('fileuploadprogressall', function (e, data) {
    var progress = parseInt(data.loaded / data.total * 100, 10);
    var strProgress = progress + "%";
    $('#fileUploadProgress .progress-bar').css({"width": strProgress});
    $('#fileUploadProgress .progress-bar').text(strProgress);
    if(progress == 100){
         $('#fileUploadProgress .progress-bar').addClass('progress-bar-striped progress-bar-animated active');
         $('#fileUploadProgress .progress-bar').text('processing..');
      }
   })
  .bind('fileuploaddone', function(e, data) {
    var error = false;
    $('#fileUploadProgress').addClass('hidden');
    $.each(data.result.files, function(index, file) {
      if (file.error) {
        error = true;
        $.notify({
                icon: 'fas fa-thumbs-down',
                message: file.error,
                },{
                type: 'danger',
                });
      }
    })
    $.post(data.url + 'done/')
      .done(setTimeout(function() { location.reload();}, 1000));
    files = [];
  })
  .bind('fileuploadfail', function(e, data) {
    $.notify({
            icon: 'fas fa-thumbs-down',
            message: 'File upload failed',
            },{
            type: 'danger',
            });
  });

});
