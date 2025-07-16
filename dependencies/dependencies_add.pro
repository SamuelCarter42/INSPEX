pro dependencies_add,new_path
  vern           = !VERSION.RELEASE       ;;  e.g., '7.1.1'

  ;;  Test to see if after version 7.0.3 and/or after version 6.2
  test__62       = (vern[0] GE '6.2') AND (vern[0] LT '7.0.3')
  test_703       = (vern[0] GE '7.0.3')
  IF (test__62) THEN PREF_SET,'IDL_PATH',new_path[0],/COMMIT
  IF (test_703) THEN !PATH = EXPAND_PATH(new_path[0])
  
end