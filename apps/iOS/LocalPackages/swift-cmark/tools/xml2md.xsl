<?xml version="1.0" encoding="UTF-8"?>

<!--

xml2md.xsl
==========

This XSLT stylesheet transforms the cmark XML format back to Commonmark.
Since the XML output is lossy, a lossless MD->XML->MD roundtrip isn't
possible. The XML->MD->XML roundtrip should produce the original XML,
though.

Example usage with xsltproc:

    cmark -t xml doc.md | xsltproc -novalid xml2md.xsl -

-->

<xsl:stylesheet
    version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:md="http://commonmark.org/xml/1.0">

<xsl:output method="text" encoding="utf-8"/>

<!-- Generic templates -->

<xsl:template match="/ | md:document | md:list">
    <xsl:apply-templates select="md:*"/>
</xsl:template>

<xsl:template match="md:*">
    <xsl:message>Unsupported element '<xsl:value-of select="local-name()"/>'</xsl:message>
</xsl:template>

<xsl:template match="md:*" mode="indent"/>

<!-- Indent blocks -->

<xsl:template match="md:*" mode="indent-block">
    <xsl:if test="preceding-sibling::md:*">
        <xsl:if test="not(ancestor::md:list[1][@tight='true'])">
            <xsl:apply-templates select="ancestor::md:*" mode="indent"/>
            <xsl:text>&#10;</xsl:text>
        </xsl:if>
        <xsl:apply-templates select="ancestor::md:*" mode="indent"/>
    </xsl:if>
</xsl:template>

<!-- Heading -->

<xsl:template match="md:heading">
    <xsl:apply-templates select="." mode="indent-block"/>
    <xsl:value-of select="substring('###### ', 7 - @level)"/>
    <xsl:apply-templates select="md:*"/>
    <xsl:text>&#10;</xsl:text>
</xsl:template>

<!-- Paragraph -->

<xsl:template match="md:paragraph">
    <xsl:apply-templates select="." mode="indent-block"/>
    <xsl:apply-templates select="md:*"/>
    <xsl:text>&#10;</xsl:text>
</xsl:template>

<!-- Thematic break -->

<xsl:template match="md:thematic_break">
    <xsl:apply-templates select="." mode="indent-block"/>
    <xsl:text>***&#10;</xsl:text>
</xsl:template>

<!-- List -->

<xsl:template match="md:list">
    <xsl:apply-templates select="." mode="indent-block"/>
    <xsl:apply-templates select="md:*"/>
</xsl:template>

<xsl:template match="md:item">
    <xsl:apply-templates select="." mode="indent-block"/>
    <xsl:choose>
        <xsl:when test="../@type = 'bullet'">-</xsl:when>
        <xsl:when test="../@type = 'ordered'">
            <xsl:value-of select="../@start + position() - 1"/>
            <xsl:choose>
                <xsl:when test="../@delim = 'period'">.</xsl:when>
                <xsl:when test="../@delim = 'paren'">)</xsl:when>
            </xsl:choose>
        </xsl:when>
    </xsl:choose>
    <xsl:text> </xsl:text>
    <xsl:apply-templates select="md:*"/>
</xsl:template>

<xsl:template match="md:item" mode="indent">
    <xsl:choose>
        <xsl:when test="../@type = 'bullet'">
            <xsl:text>  </xsl:text>
        </xsl:when>
        <xsl:when test="../@type = 'ordered'">
            <xsl:text>   </xsl:text>
        </xsl:when>
    </xsl:choose>
</xsl:template>

<!-- Block quote -->

<xsl:template match="md:block_quote">
    <xsl:apply-templates select="." mode="indent-block"/>
    <xsl:text>&gt; </xsl:text>
    <xsl:apply-templates select="md:*"/>
</xsl:template>

<xsl:template match="md:block_quote" mode="indent">
    <xsl:text>&gt; </xsl:text>
</xsl:template>

<!-- Code block -->

<xsl:template match="md:code_block">
    <xsl:apply-templates select="." mode="indent-block"/>

    <xsl:variable name="t" select="string(.)"/>
    <xsl:variable name="delim">
        <xsl:call-template name="code-delim">
            <xsl:with-param name="text" select="$t"/>
            <xsl:with-param name="delim" select="'```'"/>
        </xsl:call-template>
    </xsl:variable>

    <xsl:value-of select="$delim"/>
    <xsl:value-of select="@info"/>
    <xsl:text>&#10;</xsl:text>
    <xsl:call-template name="indent-lines">
        <xsl:with-param name="code" select="$t"/>
    </xsl:call-template>
    <xsl:apply-templates select="ancestor::md:*" mode="indent"/>
    <xsl:value-of select="$delim"/>
    <xsl:text>&#10;</xsl:text>
</xsl:template>

<!-- Inline HTML -->

<xsl:template match="md:html_block">
    <xsl:apply-templates select="." mode="indent-block"/>
    <xsl:value-of select="substring-before(., '&#10;')"/>
    <xsl:text>&#10;</xsl:text>
    <xsl:call-template name="indent-lines">
        <xsl:with-param name="code" select="substring-after(., '&#10;')"/>
    </xsl:call-template>
</xsl:template>

<!-- Indent multiple lines -->

<xsl:template name="indent-lines">
    <xsl:param name="code"/>
    <xsl:if test="contains($code, '&#10;')">
        <xsl:apply-templates select="ancestor::md:*" mode="indent"/>
        <xsl:value-of select="substring-before($code, '&#10;')"/>
        <xsl:text>&#10;</xsl:text>
        <xsl:call-template name="indent-lines">
            <xsl:with-param name="code" select="substring-after($code, '&#10;')"/>
        </xsl:call-template>
    </xsl:if>
</xsl:template>

<!-- Text -->

<xsl:template match="md:text">
    <xsl:variable name="t" select="string(.)"/>
    <xsl:variable name="first" select="substring($t, 1, 1)"/>
    <xsl:variable name="marker-check" select="translate(substring($t, 1, 10), '0123456789', '')"/>
    <xsl:choose>
        <!-- Escape ordered list markers -->
        <xsl:when test="starts-with($marker-check, '.') and $first != '.'">
            <xsl:value-of select="substring-before($t, '.')"/>
            <xsl:text>\.</xsl:text>
            <xsl:call-template name="escape-text">
                <xsl:with-param name="text" select="substring-after($t, '.')"/>
            </xsl:call-template>
        </xsl:when>
        <xsl:when test="starts-with($marker-check, ')') and $first != ')'">
            <xsl:value-of select="substring-before($t, ')')"/>
            <xsl:text>\)</xsl:text>
            <xsl:call-template name="escape-text">
                <xsl:with-param name="text" select="substring-after($t, ')')"/>
            </xsl:call-template>
        </xsl:when>
        <!-- Escape leading block characters -->
        <xsl:when test="contains('-+>#=~', $first)">
            <xsl:text>\</xsl:text>
            <xsl:value-of select="$first"/>
            <xsl:call-template name="escape-text">
                <xsl:with-param name="text" select="substring($t, 2)"/>
            </xsl:call-template>
        </xsl:when>
        <!-- Otherwise -->
        <xsl:otherwise>
            <xsl:call-template name="escape-text">
                <xsl:with-param name="text" select="$t"/>
            </xsl:call-template>
        </xsl:otherwise>
    </xsl:choose>
</xsl:template>

<!-- Breaks -->

<xsl:template match="md:softbreak">
    <xsl:text>&#10;</xsl:text>
    <xsl:apply-templates select="ancestor::md:*" mode="indent"/>
</xsl:template>

<xsl:template match="md:linebreak">
    <xsl:text>  &#10;</xsl:text>
    <xsl:apply-templates select="ancestor::md:*" mode="indent"/>
</xsl:template>

<!-- Emphasis -->

<xsl:template match="md:emph">
    <xsl:text>*</xsl:text>
    <xsl:apply-templates select="md:*"/>
    <xsl:text>*</xsl:text>
</xsl:template>

<xsl:template match="md:strong">
    <xsl:text>**</xsl:text>
    <xsl:apply-templates select="md:*"/>
    <xsl:text>**</xsl:text>
</xsl:template>

<!-- Inline code -->

<xsl:template match="md:code">
    <xsl:variable name="t" select="string(.)"/>
    <xsl:variable name="delim">
        <xsl:call-template name="code-delim">
            <xsl:with-param name="text" select="$t"/>
            <xsl:with-param name="delim" select="'`'"/>
        </xsl:call-template>
    </xsl:variable>
    <xsl:value-of select="$delim"/>
    <xsl:value-of select="$t"/>
    <xsl:value-of select="$delim"/>
</xsl:template>

<!-- Links and images -->

<xsl:template match="md:link | md:image">
    <xsl:if test="self::md:image">!</xsl:if>
    <xsl:text>[</xsl:text>
    <xsl:apply-templates select="md:*"/>
    <xsl:text>](</xsl:text>
    <xsl:call-template name="escape-text">
        <xsl:with-param name="text" select="string(@destination)"/>
        <xsl:with-param name="escape" select="'()'"/>
    </xsl:call-template>
    <xsl:if test="string(@title)">
        <xsl:text> "</xsl:text>
        <xsl:call-template name="escape-text">
            <xsl:with-param name="text" select="string(@title)"/>
            <xsl:with-param name="escape" select="'&quot;'"/>
        </xsl:call-template>
        <xsl:text>"</xsl:text>
    </xsl:if>
    <xsl:text>)</xsl:text>
</xsl:template>

<!-- Inline HTML -->

<xsl:template match="md:html_inline">
    <xsl:value-of select="."/>
</xsl:template>

<!-- Escaping helpers -->

<xsl:template name="escape-text">
    <xsl:param name="text"/>
    <xsl:param name="escape" select="'*_`&lt;[]&amp;'"/>

    <xsl:variable name="trans" select="translate($text, $escape, '\\\\\\\')"/>
    <xsl:choose>
        <xsl:when test="contains($trans, '\')">
            <xsl:variable name="safe" select="substring-before($trans, '\')"/>
            <xsl:variable name="l" select="string-length($safe)"/>
            <xsl:value-of select="$safe"/>
            <xsl:text>\</xsl:text>
            <xsl:value-of select="substring($text, $l + 1, 1)"/>
            <xsl:call-template name="escape-text">
                <xsl:with-param name="text" select="substring($text, $l + 2)"/>
                <xsl:with-param name="escape" select="$escape"/>
            </xsl:call-template>
        </xsl:when>
        <xsl:otherwise>
            <xsl:value-of select="$text"/>
        </xsl:otherwise>
    </xsl:choose>
</xsl:template>

<xsl:template name="code-delim">
    <xsl:param name="text"/>
    <xsl:param name="delim"/>

    <xsl:choose>
        <xsl:when test="contains($text, $delim)">
            <xsl:call-template name="code-delim">
                <xsl:with-param name="text" select="$text"/>
                <xsl:with-param name="delim" select="concat($delim, '`')"/>
            </xsl:call-template>
        </xsl:when>
        <xsl:otherwise>
            <xsl:value-of select="$delim"/>
        </xsl:otherwise>
    </xsl:choose>
</xsl:template>

</xsl:stylesheet>
