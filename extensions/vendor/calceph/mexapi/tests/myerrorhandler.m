function myerrorhandler (message)
       global myfunccall
       myfunccall = 1;
       'mycustom error function '
       disp(message);
end
