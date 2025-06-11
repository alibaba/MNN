list(APPEND _dict_compile_list
  af am an ar as az
  ba be bg bn bpy bs
  ca chr cmn cs cv cy
  da de
  el en eo es et eu
  fa fi fr
  ga gd gn grc gu
  hak haw he hi hr ht hu hy
  ia id io is it
  ja jbo
  ka kk kl kn kok ko ku ky
  la lb lfn lt lv
  mi mk ml mr ms mto mt my
  nci ne nl nog no
  om or
  pap pa piqd pl pt py
  qdb quc qu qya
  ro ru
  sd shn si sjn sk sl smj sq sr sv sw
  ta te ti th tk tn tr tt
  ug uk ur uz
  vi
  yue
)

list(APPEND _mbrola_lang_list
  af1 ar1 ar2
  ca cmn cr1 cs
  de2 de4 de6 de8
  ee1 en1 es es3 es4
  fr
  gr1 gr2 grc-de6
  he hn1 hu1
  ic1 id1 in ir1 it1 it3
  jp
  la1 lt
  ma1 mx1 mx2
  nl nz1
  pl1 pt1 ptbr ptbr4
  ro1
  sv sv2
  tl1 tr1
  us us3
  vz
)

set(DATA_DIST_ROOT ${CMAKE_CURRENT_BINARY_DIR})
set(DATA_DIST_DIR ${DATA_DIST_ROOT}/espeak-ng-data)
set(PHONEME_TMP_DIR ${DATA_DIST_ROOT}/phsource)
set(DICT_TMP_DIR ${DATA_DIST_ROOT}/dictsource)

set(DATA_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/espeak-ng-data)
set(PHONEME_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/phsource)
set(DICT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dictsource)

file(MAKE_DIRECTORY "${DATA_DIST_DIR}")
file(MAKE_DIRECTORY "${DICT_TMP_DIR}")
file(COPY "${DATA_SRC_DIR}/lang" DESTINATION "${DATA_DIST_DIR}")
file(COPY "${DATA_SRC_DIR}/voices/!v" DESTINATION "${DATA_DIST_DIR}/voices")
file(COPY "${PHONEME_SRC_DIR}" DESTINATION "${DATA_DIST_ROOT}")

set(ESPEAK_RUN_ENV ${CMAKE_COMMAND} -E env "ESPEAK_DATA_PATH=${DATA_DIST_ROOT}")
set(ESPEAK_RUN_CMD ${ESPEAK_RUN_ENV} $ENV{VALGRIND} "$<TARGET_FILE:espeak-ng-bin>")

add_custom_command(
  OUTPUT "${DATA_DIST_DIR}/intonations"
  COMMAND ${ESPEAK_RUN_CMD} --compile-intonations
  WORKING_DIRECTORY "${PHONEME_SRC_DIR}"
  COMMENT "Compile intonations"
  DEPENDS
    "$<TARGET_FILE:espeak-ng-bin>"
    "${PHONEME_SRC_DIR}/intonation"
)

set(_phon_deps "")

function(check_phon_deps _file)
  set(_file "${PHONEME_SRC_DIR}/${_file}")
  list(APPEND _phon_deps "${_file}")

  file(STRINGS "${_file}" _phon_incs REGEX "include .+")
  list(TRANSFORM _phon_incs REPLACE "^[ \t]*include[ \t]+" "")
  foreach(_inc ${_phon_incs})
    check_phon_deps(${_inc})
  endforeach(_inc)
  set(_phon_deps ${_phon_deps} PARENT_SCOPE)
endfunction(check_phon_deps)

check_phon_deps("phonemes")

add_custom_command(
  OUTPUT
    "${DATA_DIST_DIR}/phondata"
    "${DATA_DIST_DIR}/phondata-manifest"
    "${DATA_DIST_DIR}/phonindex"
    "${DATA_DIST_DIR}/phontab"
  COMMAND ${ESPEAK_RUN_CMD} --compile-phonemes
  WORKING_DIRECTORY "${PHONEME_SRC_DIR}"
  COMMENT "Compile phonemes"
  DEPENDS
    "${DATA_DIST_DIR}/intonations"
    "$<TARGET_FILE:espeak-ng-bin>"
    ${_phon_deps}
)

list(APPEND _dict_targets)
list(APPEND _mbr_targets)

foreach(_dict_name ${_dict_compile_list})
  set(_dict_target "${DATA_DIST_DIR}/${_dict_name}_dict")
  set(_dict_deps "")
  list(APPEND _dict_targets ${_dict_target})
  list(APPEND _dict_deps
    "${DICT_SRC_DIR}/${_dict_name}_rules"
    "${DICT_SRC_DIR}/${_dict_name}_list"
  )

  if(EXISTS "${DICT_SRC_DIR}/extra/${_dict_name}_listx")
    option(EXTRA_${_dict_name} "Compile extra ${_dict_name} dictionary" ON)
    if(EXTRA_${_dict_name})
      list(APPEND _dict_deps "${DICT_SRC_DIR}/extra/${_dict_name}_listx")
    else()
      file(REMOVE "${DICT_TMP_DIR}/${_dict_name}_listx")
    endif()
  elseif(EXISTS "${DICT_SRC_DIR}/${_dict_name}_listx")
    list(APPEND _dict_deps "${DICT_SRC_DIR}/${_dict_name}_listx")
  endif()
  if(EXISTS "${DICT_SRC_DIR}/${_dict_name}_emoji")
    list(APPEND _dict_deps "${DICT_SRC_DIR}/${_dict_name}_emoji")
  endif()

  add_custom_command(
    OUTPUT "${_dict_target}"
    COMMAND ${CMAKE_COMMAND} -E copy ${_dict_deps} "${DICT_TMP_DIR}/"
    COMMAND ${ESPEAK_RUN_CMD} --compile=${_dict_name}
    WORKING_DIRECTORY "${DICT_TMP_DIR}"
    DEPENDS
      "$<TARGET_FILE:espeak-ng-bin>"
      "${DATA_DIST_DIR}/phondata"
      "${DATA_DIST_DIR}/intonations"
      ${_dict_deps}
  )
endforeach()

if (HAVE_MBROLA AND USE_MBROLA)
  file(COPY "${DATA_SRC_DIR}/voices/mb" DESTINATION "${DATA_DIST_DIR}/voices")
  file(MAKE_DIRECTORY "${DATA_DIST_DIR}/mbrola_ph")
  foreach(_mbl ${_mbrola_lang_list})
    set(_mbl_src "${PHONEME_SRC_DIR}/mbrola/${_mbl}")
    set(_mbl_out "${DATA_DIST_DIR}/mbrola_ph/${_mbl}_phtrans")
    list(APPEND _mbr_targets ${_mbl_out})
    add_custom_command(
      OUTPUT "${_mbl_out}"
      COMMAND ${ESPEAK_RUN_CMD} --compile-mbrola="${_mbl_src}"
      DEPENDS "$<TARGET_FILE:espeak-ng-bin>" "${_mbl_src}"
    )
  endforeach(_mbl)
endif()

add_custom_target(
  data ALL
  DEPENDS
    "${DATA_DIST_DIR}/intonations"
    "${DATA_DIST_DIR}/phondata"
    ${_dict_targets}
    ${_mbr_targets}
)
install(DIRECTORY ${DATA_DIST_DIR} DESTINATION share)
