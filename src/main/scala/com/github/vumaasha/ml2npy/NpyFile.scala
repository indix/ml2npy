package com.github.vumaasha.ml2npy

import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import java.nio.file.{Paths, StandardOpenOption}
import java.nio.{ByteBuffer, ByteOrder}
import scala.reflect.runtime.universe._


/**
  * Created by vumaasha on 24/12/16.
  */
abstract class NpyFile[V] {

  val magic: Array[Byte] = 0X93.toByte +: "NUMPY".getBytes(StandardCharsets.US_ASCII)
  val majorVersion = 1
  val minorVersion = 0
  val headerLenth = 70
  /*
  supported dtypes i2,i4,i8,f4,f8
   */
  val dtype: String
  val dataSize: Int

  def addElements(data: Seq[V]) = {
    val dataLength = data.length.toString
    val description = s"{'descr': '$dtype', 'fortran_order': False, 'shape': ($dataLength,), }"

    val versionByteSize = 2
    val headerByteSize = 2
    val newLineSize = 1
    val unpaddedLength: Int = magic.length + versionByteSize + headerByteSize + description.length + newLineSize
    val isPaddingRequired = (unpaddedLength % 16) != 0
    val paddingLength = (((unpaddedLength / 16) + 1) * 16) - unpaddedLength
    val paddedDescription = if (isPaddingRequired) {
      description + (" " * paddingLength)
    } else description

    val headerLength: Int = unpaddedLength + paddingLength
    val content = ByteBuffer.allocateDirect(headerLength + (data.length * dataSize))
      .order(ByteOrder.LITTLE_ENDIAN)
      .put(magic)
      .put(majorVersion.toByte)
      .put(minorVersion.toByte)
      .putShort(paddedDescription.length.toShort)
      .put(paddedDescription.getBytes(StandardCharsets.US_ASCII))

    data.foreach(addElement(content))
    content.flip
    content
  }

  def addElement(content: ByteBuffer)(elem: V): ByteBuffer
}

class LongNpyFile extends NpyFile[Long] {
  override val dtype: String = "<i8"
  override val dataSize: Int = 8

  override def addElement(content: ByteBuffer)(elem: Long): ByteBuffer = content.putLong(elem)
}

class IntNpyFile extends NpyFile[Int] {
  override val dtype: String = "<i4"
  override val dataSize: Int = 4

  override def addElement(content: ByteBuffer)(elem: Int) = content.putInt(elem)
}

class ShortNpyFile extends NpyFile[Short] {
  override val dtype: String = "<i2"
  override val dataSize: Int = 2

  override def addElement(content: ByteBuffer)(elem: Short) = content.putLong(elem)
}

class ByteNpyFile extends NpyFile[Byte] {
  override val dtype: String = "<i1"
  override val dataSize: Int = 1

  override def addElement(content: ByteBuffer)(elem: Byte) = content.putLong(elem)
}

class FloatNpyFile extends NpyFile[Float] {
  override val dtype: String = "<f4"
  override val dataSize: Int = 4

  override def addElement(content: ByteBuffer)(elem: Float) = content.putFloat(elem)
}

class DoubleNpyFile extends NpyFile[Double] {
  override val dtype: String = "<f8"
  override val dataSize: Int = 8

  override def addElement(content: ByteBuffer)(elem: Double) = content.putDouble(elem)
}


object NpyTester {
  def main(args: Array[String]): Unit = {
    /*
    To test the export start ipython and run following commands
    import numpy as np
    np.load('/tmp/test.npy')
    The imported matrix should contain the values (0.3,0.5)
     */
    val floatNpyFile:FloatNpyFile = new FloatNpyFile
    val content = floatNpyFile.addElements(Seq(0.3f,0.5f))
    val channel = FileChannel.open(Paths.get("/tmp/test.npy"), StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE)
    channel.write(content)
    channel.close()

    val intNpyFile:IntNpyFile = new IntNpyFile
    val intContent = intNpyFile.addElements(1 to 10)
    val intChannel = FileChannel.open(Paths.get("/tmp/inttest.npy"), StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE)
    intChannel.write(intContent)
    intChannel.close()

  }
}