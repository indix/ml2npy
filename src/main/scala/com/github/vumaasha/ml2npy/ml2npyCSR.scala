package com.github.vumaasha.ml2npy

import java.nio.ByteBuffer

/**
  * Created by vumaasha on 25/12/16.
  */

class ml2npyCSR(data: Seq[Float], indices: Seq[Int], indexPointers: Seq[Int], rows: Int, columns: Int) {
  val dataB: ByteBuffer = (new FloatNpyFile).addElements(data)
  val indicesB: ByteBuffer = (new IntNpyFile).addElements(indices)
  val indexPointersB: ByteBuffer = (new IntNpyFile).addElements(indexPointers)
  val shapeB: ByteBuffer = (new IntNpyFile).addElements(Array(rows,columns))

}
