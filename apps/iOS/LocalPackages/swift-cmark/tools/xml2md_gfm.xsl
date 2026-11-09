<?xml version="1.0" encoding="UTF-8"?>

<!--

xml2md_gfm.xsl
==============

This XSLT stylesheet is a complement to xml2md.xsl with templates supporting GitHub-flavored Markdown extensions (tables, strike-through).

-->

<xsl:stylesheet
    version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:md="http://commonmark.org/xml/1.0">


<!-- Import commonmark XSL -->

<xsl:import href="xml2md.xsl"/>
<xsl:template match="/">
  <xsl:apply-imports/>
</xsl:template>

<!-- params -->

<xsl:output method="text" encoding="utf-8"/>


<!-- Table -->

<xsl:template match="md:table">
    <xsl:apply-templates select="." mode="indent-block"/>
    <xsl:apply-templates select="md:*"/>
</xsl:template>

<xsl:template match="md:table_header">
    <xsl:text>| </xsl:text>
    <xsl:apply-templates select="md:*"/>
    <xsl:text>&#xa; | </xsl:text>
     <xsl:for-each select="md:table_cell">
     <xsl:choose>
     <xsl:when test="@align = 'right'">
     <xsl:text> ---: |</xsl:text>
     </xsl:when>
     <xsl:when test="@align = 'left'">
     <xsl:text> :--- |</xsl:text>
     </xsl:when>
     <xsl:when test="@align = 'center'">
     <xsl:text> :---: |</xsl:text>
     </xsl:when>
     <xsl:otherwise>
     <xsl:text> --- |</xsl:text>
     </xsl:otherwise>
     </xsl:choose>
     </xsl:for-each>
    <xsl:text>&#xa;</xsl:text>
</xsl:template>

<xsl:template match="md:table_cell">
    <xsl:apply-templates select="md:*"/>
    <xsl:text>| </xsl:text>
</xsl:template>

<xsl:template match="md:table_row">
    <xsl:text>| </xsl:text>
    <xsl:apply-templates select="md:*"/>
    <xsl:text>&#xa;</xsl:text>
</xsl:template>


<!-- Striked-through -->

<xsl:template match="md:strikethrough">
    <xsl:text>~~</xsl:text>
    <xsl:apply-templates select="md:*"/>
    <xsl:text>~~</xsl:text>
</xsl:template>

</xsl:stylesheet>
