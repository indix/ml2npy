package com.github.vumaasha.ml2npy

/**
  * Created by vumaasha on 25/12/16.
  */

import scala.{specialized => spec}

class ml2npyCSR[@spec(Double, Int, Float, Long, Short, Byte) V](data: Seq[V],
                                                                indices: Seq[Int],
                                                                indexPointers: Seq[Int],
                                                                rows: Int,
                                                                columns: Int) {

}
