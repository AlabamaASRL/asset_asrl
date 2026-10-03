This function defines the behavior of the library when an error occurs during the execution of the library's functions. This function should be (not mandatory) called before any other functions of the library. The behavior depends on the value of *typehandler*. 

The possible values for *typehandler* are  :

+-------+---------------------------------------------------------------------------------------+
| value | meaning                                                                               |
+=======+=======================================================================================+
| 1     | | The library displays a message and continues the execution.                         |
|       | | The functions return an error code. The python and Octave/Matlab interfaces raise   |
|       | | an exception.                                                                       |
|       | | This is the default behavior of the library.                                        |
+-------+---------------------------------------------------------------------------------------+
| 2     | | The library displays a message                                                      |
|       | | and terminates the execution with a system call to the function *exit*.             |
+-------+---------------------------------------------------------------------------------------+
| 3     | | The library calls the user function *userfunc* with the message.                    |
+-------+---------------------------------------------------------------------------------------+