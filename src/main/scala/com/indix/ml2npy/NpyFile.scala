package com.indix.ml2npy

import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import java.nio.file.{Paths, StandardOpenOption}
import java.nio.{ByteBuffer, ByteOrder}
import scala.collection.immutable.Range.Inclusive
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer


/**
  * Created by vumaasha on 24/12/16.
  */
abstract class NpyFile[V] {

  /*
  supported dtypes i2,i4,i8,f4,f8
   */
  val dtype: String
  val dataSize: Int
  val dataBuffer: ArrayBuffer[Byte] = ArrayBuffer.empty[Byte]
  var numElements:Int = 0

  def getBytes:Array[Byte] = {
    require(numElements >0, "Elements should be added to the buffer before getting bytes")
    (getHeader(numElements) ++ dataBuffer).toArray
  }

  def getHeader(dataLength: Int): ArrayBuffer[Byte] = {

    val magic: Array[Byte] = 0X93.toByte +: "NUMPY".getBytes(StandardCharsets.US_ASCII)
    val majorVersion = 1
    val minorVersion = 0

    val header: ArrayBuffer[Byte] = ArrayBuffer.empty[Byte]
    val description = s"{'descr': '$dtype', 'fortran_order': False, 'shape': ($dataLength,), }"

    val versionByteSize = 2
    val headerByteSize = 2
    val newLineSize = 1
    val unpaddedLength: Int = magic.length + versionByteSize + headerByteSize + description.length + newLineSize
    val isPaddingRequired = (unpaddedLength % 16) != 0
    val paddingLength = 16 - (unpaddedLength % 16)
    val paddedDescription = {
      if (isPaddingRequired) {
        description + (" " * paddingLength)
      } else description
    } + "\n"

    val headerLength: Int = unpaddedLength + paddingLength
    val content = ByteBuffer.allocate(headerLength)
      .order(ByteOrder.LITTLE_ENDIAN)
      .put(magic)
      .put(majorVersion.toByte)
      .put(minorVersion.toByte)
      .putShort(paddedDescription.length.toShort)
      .put(paddedDescription.getBytes(StandardCharsets.US_ASCII))

    header ++ content.array()
  }


  def addElements(data: Seq[V]) = {
    val header = getHeader(data.length)
    val headerLength: Int = header.length
    val content = ByteBuffer.allocate(headerLength + (data.length * dataSize))
      .order(ByteOrder.LITTLE_ENDIAN)
      .put(header.toArray)

    data.foreach(addElement(content))
    content.flip
    content
  }

  protected def addElement(content: ByteBuffer)(elem: V): ByteBuffer

  def addToBuffer(elem:V):Unit = {
    val buffer: ByteBuffer = ByteBuffer.allocate(dataSize).order(ByteOrder.LITTLE_ENDIAN)
    val elemBytes: mutable.ArrayOps[Byte] = addElement(buffer)(elem).array()
    dataBuffer ++= elemBytes
    numElements += 1
  }
}

object NpyFile {
  trait NpyFileFactory[V] {
    def create(): NpyFile[V]
  }

  class IntNpyFile extends NpyFile[Int] {
    override val dtype: String = "<i4"
    override val dataSize: Int = 4

    override def addElement(content: ByteBuffer)(elem: Int) = content.putInt(elem)
  }

  implicit object IntNpyFile extends NpyFileFactory[Int] {
    def create() = new IntNpyFile()
  }

  class LongNpyFile extends NpyFile[Long] {
    override val dtype: String = "<i8"
    override val dataSize: Int = 8

    override def addElement(content: ByteBuffer)(elem: Long): ByteBuffer = content.putLong(elem)
  }

  implicit object LongNpyFile extends NpyFileFactory[Long] {
    def create() = new LongNpyFile()
  }

  class ShortNpyFile extends NpyFile[Short] {
    override val dtype: String = "<i2"
    override val dataSize: Int = 2

    override def addElement(content: ByteBuffer)(elem: Short) = content.putShort(elem)
  }

  implicit object ShortNpyFile extends NpyFileFactory[Short] {
    def create() = new ShortNpyFile()
  }

  class ByteNpyFile extends NpyFile[Byte] {
    override val dtype: String = "<i1"
    override val dataSize: Int = 1

    override def addElement(content: ByteBuffer)(elem: Byte) = content.put(elem)
  }

  implicit object ByteNpyFile  extends NpyFileFactory[Byte] {
    def create() = new ByteNpyFile()
  }

  class FloatNpyFile extends NpyFile[Float] {
    override val dtype: String = "<f4"
    override val dataSize: Int = 4

    override def addElement(content: ByteBuffer)(elem: Float) = content.putFloat(elem)
  }

  implicit object FloatNpyFile extends NpyFileFactory[Float] {
    def create() = new FloatNpyFile()
  }

  class DoubleNpyFile extends NpyFile[Double] {
    override val dtype: String = "<f8"
    override val dataSize: Int = 8

    override def addElement(content: ByteBuffer)(elem: Double) = content.putDouble(elem)
  }

  implicit object DoubleNpyFile extends NpyFileFactory[Double] {
    def create() = new DoubleNpyFile()
  }

  def apply[V]()(implicit ev: NpyFileFactory[V]): NpyFile[V] = ev.create()
}
