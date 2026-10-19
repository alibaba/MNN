#[[
This source file is part of the Swift Collections Open Source Project

Copyright (c) 2022 - 2024 Apple Inc. and the Swift project authors
Licensed under Apache License v2.0 with Runtime Library Exception

See https://swift.org/LICENSE.txt for license information
#]]

if(COLLECTIONS_SINGLE_MODULE)
  set(module_name ${COLLECTIONS_MODULE_NAME})
else()
  set(module_name BitCollections)
  add_library(BitCollections
    ${COLLECTIONS_BIT_SOURCES})
  target_link_libraries(BitCollections PRIVATE
    InternalCollectionsUtilities)
  set_target_properties(BitCollections PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})

  _install_target(BitCollections)
  set_property(GLOBAL APPEND PROPERTY SWIFT_COLLECTIONS_EXPORTS BitCollections)
endif()

target_sources(${module_name} PRIVATE
  "BitArray/BitArray+BitwiseOperations.swift"
  "BitArray/BitArray+ChunkedBitsIterators.swift"
  "BitArray/BitArray+Codable.swift"
  "BitArray/BitArray+Collection.swift"
  "BitArray/BitArray+Copy.swift"
  "BitArray/BitArray+CustomReflectable.swift"
  "BitArray/BitArray+Descriptions.swift"
  "BitArray/BitArray+Equatable.swift"
  "BitArray/BitArray+ExpressibleByArrayLiteral.swift"
  "BitArray/BitArray+ExpressibleByStringLiteral.swift"
  "BitArray/BitArray+Extras.swift"
  "BitArray/BitArray+Fill.swift"
  "BitArray/BitArray+Hashable.swift"
  "BitArray/BitArray+Initializers.swift"
  "BitArray/BitArray+Invariants.swift"
  "BitArray/BitArray+LosslessStringConvertible.swift"
  "BitArray/BitArray+RandomBits.swift"
  "BitArray/BitArray+RangeReplaceableCollection.swift"
  "BitArray/BitArray+Testing.swift"
  "BitArray/BitArray._UnsafeHandle.swift"
  "BitArray/BitArray.swift"
  "BitSet/BitSet+BidirectionalCollection.swift"
  "BitSet/BitSet+Codable.swift"
  "BitSet/BitSet+CustomDebugStringConvertible.swift"
  "BitSet/BitSet+CustomReflectable.swift"
  "BitSet/BitSet+CustomStringConvertible.swift"
  "BitSet/BitSet+Equatable.swift"
  "BitSet/BitSet+ExpressibleByArrayLiteral.swift"
  "BitSet/BitSet+Extras.swift"
  "BitSet/BitSet+Hashable.swift"
  "BitSet/BitSet+Initializers.swift"
  "BitSet/BitSet+Invariants.swift"
  "BitSet/BitSet+Random.swift"
  "BitSet/BitSet+SetAlgebra basics.swift"
  "BitSet/BitSet+SetAlgebra conformance.swift"
  "BitSet/BitSet+SetAlgebra formIntersection.swift"
  "BitSet/BitSet+SetAlgebra formSymmetricDifference.swift"
  "BitSet/BitSet+SetAlgebra formUnion.swift"
  "BitSet/BitSet+SetAlgebra intersection.swift"
  "BitSet/BitSet+SetAlgebra isDisjoint.swift"
  "BitSet/BitSet+SetAlgebra isEqualSet.swift"
  "BitSet/BitSet+SetAlgebra isStrictSubset.swift"
  "BitSet/BitSet+SetAlgebra isStrictSuperset.swift"
  "BitSet/BitSet+SetAlgebra isSubset.swift"
  "BitSet/BitSet+SetAlgebra isSuperset.swift"
  "BitSet/BitSet+SetAlgebra subtract.swift"
  "BitSet/BitSet+SetAlgebra subtracting.swift"
  "BitSet/BitSet+SetAlgebra symmetricDifference.swift"
  "BitSet/BitSet+SetAlgebra union.swift"
  "BitSet/BitSet+Sorted Collection APIs.swift"
  "BitSet/BitSet.Counted.swift"
  "BitSet/BitSet.Index.swift"
  "BitSet/BitSet._UnsafeHandle.swift"
  "BitSet/BitSet.swift"
  "Shared/Range+Utilities.swift"
  "Shared/Slice+Utilities.swift"
  "Shared/UInt+Tricks.swift"
  "Shared/_Word.swift"
  )
