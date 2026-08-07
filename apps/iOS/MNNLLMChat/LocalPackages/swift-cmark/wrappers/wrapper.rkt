#lang racket/base

;; requires racket >= 5.3 because of submodules

;; Lowlevel interface

(module low-level racket/base

  (require ffi/unsafe ffi/unsafe/define)

  (provide (all-defined-out))

  (define-ffi-definer defcmark (ffi-lib "libcmark"))

  (define _cmark_node_type
    (_enum '(;; Error status
             none
             ;; Block
             document block-quote list item code-block
             html-block custom-block
             paragraph heading thematic-break
             ;; ?? first-block = document
             ;; ?? last-block = thematic-break
             ;; Inline
             text softbreak linebreak code html-inline custom-inline
             emph strong link image
             ;; ?? first-inline = text
             ;; ?? last-inline = image
             )))
  (define _cmark_list_type
    (_enum '(no_list bullet_list ordered_list)))
  (define _cmark_delim_type
    (_enum '(no_delim period_delim paren_delim)))
  (define _cmark_opts
    (_bitmask
     '(sourcepos  = 2 ; include sourcepos attribute on block elements
       hardbreaks = 4 ; render `softbreak` elements as hard line breaks
       safe       = 8 ; suppress raw HTML and unsafe links
       nobreaks   = 16 ; render `softbreak` elements as spaces
       normalize  = 256 ; legacy (no effect)
       validate-utf8 = 512 ; validate UTF-8 in the input
       smart      = 1024 ; straight quotes to curly, ---/-- to em/en dashes
       )))

  (define-cpointer-type _node)

  (defcmark cmark_markdown_to_html
    (_fun [bs : _bytes] [_int = (bytes-length bs)] _cmark_opts
          -> [r : _bytes] -> (begin0 (bytes->string/utf-8 r) (free r))))

  (defcmark cmark_parse_document
    (_fun [bs : _bytes] [_int = (bytes-length bs)] _cmark_opts
          -> _node))

  (defcmark cmark_render_html
    (_fun _node _cmark_opts
          -> [r : _bytes] -> (begin0 (bytes->string/utf-8 r) (free r))))

  (defcmark cmark_node_new              (_fun _cmark_node_type -> _node))
  (defcmark cmark_node_free             (_fun _node -> _void))

  (defcmark cmark_node_next             (_fun _node -> _node/null))
  (defcmark cmark_node_previous         (_fun _node -> _node/null))
  (defcmark cmark_node_parent           (_fun _node -> _node/null))
  (defcmark cmark_node_first_child      (_fun _node -> _node/null))
  (defcmark cmark_node_last_child       (_fun _node -> _node/null))

  (defcmark cmark_node_get_user_data    (_fun _node -> _racket))
  (defcmark cmark_node_set_user_data    (_fun _node _racket -> _bool))
  (defcmark cmark_node_get_type         (_fun _node -> _cmark_node_type))
  (defcmark cmark_node_get_type_string  (_fun _node -> _bytes))
  (defcmark cmark_node_get_literal      (_fun _node -> _string))
  (defcmark cmark_node_set_literal      (_fun _node _string -> _bool))
  (defcmark cmark_node_get_heading_level (_fun _node -> _int))
  (defcmark cmark_node_set_heading_level (_fun _node _int -> _bool))
  (defcmark cmark_node_get_list_type    (_fun _node -> _cmark_list_type))
  (defcmark cmark_node_set_list_type    (_fun _node _cmark_list_type -> _bool))
  (defcmark cmark_node_get_list_delim   (_fun _node -> _cmark_delim_type))
  (defcmark cmark_node_set_list_delim   (_fun _node _cmark_delim_type -> _bool))
  (defcmark cmark_node_get_list_start   (_fun _node -> _int))
  (defcmark cmark_node_set_list_start   (_fun _node _int -> _bool))
  (defcmark cmark_node_get_list_tight   (_fun _node -> _bool))
  (defcmark cmark_node_set_list_tight   (_fun _node _bool -> _bool))
  (defcmark cmark_node_get_fence_info   (_fun _node -> _string))
  (defcmark cmark_node_set_fence_info   (_fun _node _string -> _bool))
  (defcmark cmark_node_get_url          (_fun _node -> _string))
  (defcmark cmark_node_set_url          (_fun _node _string -> _bool))
  (defcmark cmark_node_get_title        (_fun _node -> _string))
  (defcmark cmark_node_set_title        (_fun _node _string -> _bool))
  (defcmark cmark_node_get_start_line   (_fun _node -> _int))
  (defcmark cmark_node_get_start_column (_fun _node -> _int))
  (defcmark cmark_node_get_end_line     (_fun _node -> _int))
  (defcmark cmark_node_get_end_column   (_fun _node -> _int))

  (defcmark cmark_node_unlink           (_fun _node -> _void))
  (defcmark cmark_node_insert_before    (_fun _node _node -> _bool))
  (defcmark cmark_node_insert_after     (_fun _node _node -> _bool))
  (defcmark cmark_node_prepend_child    (_fun _node _node -> _bool))
  (defcmark cmark_node_append_child     (_fun _node _node -> _bool))
  (defcmark cmark_consolidate_text_nodes (_fun _node -> _void))

  (defcmark cmark_version               (_fun -> _int))
  (defcmark cmark_version_string        (_fun -> _string))

  )

;; Rackety interface

(module high-level racket/base

  (require (submod ".." low-level) ffi/unsafe)

  (provide cmark-markdown-to-html)
  (define (cmark-markdown-to-html str [options '(normalize smart)])
    (cmark_markdown_to_html (if (bytes? str) str (string->bytes/utf-8 str))
                            options))

  (require (for-syntax racket/base racket/syntax))
  (define-syntax (make-getter+setter stx)
    (syntax-case stx ()
      [(_ name) (with-syntax ([(getter setter)
                               (map (λ(op) (format-id #'name "cmark_node_~a_~a"
                                                      op #'name))
                                    '(get set))])
                  #'(cons getter setter))]))
  (define-syntax-rule (define-getters+setters name [type field ...] ...)
    (define name (list (list 'type (make-getter+setter field) ...) ...)))
  (define-getters+setters getters+setters
    [heading heading_level] [code-block fence_info]
    [link url title] [image url title]
    [list list_type list_delim list_start list_tight])

  (provide cmark->sexpr)
  (define (cmark->sexpr node)
    (define text (cmark_node_get_literal node))
    (define type (cmark_node_get_type node))
    (define children
      (let loop ([node (cmark_node_first_child node)])
        (if (not node) '()
            (cons (cmark->sexpr node) (loop (cmark_node_next node))))))
    (define info
      (cond [(assq type getters+setters)
             => (λ(gss) (map (λ(gs) ((car gs) node)) (cdr gss)))]
            [else '()]))
    (define (assert-no what-not b)
      (when b (error 'cmark->sexpr "unexpected ~a in ~s" what-not type)))
    (cond [(memq type '(document paragraph heading block-quote list item
                        emph strong link image))
           (assert-no 'text text)
           (list type info children)]
          [(memq type '(text code code-block html-block html-inline
                        softbreak linebreak thematic-break))
           (assert-no 'children (pair? children))
           (list type info text)]
          [else (error 'cmark->sexpr "unknown type: ~s" type)]))

  (provide sexpr->cmark)
  (define (sexpr->cmark sexpr) ; assumes valid input, as generated by the above
    (define (loop sexpr)
      (define type (car sexpr))
      (define info (cadr sexpr))
      (define data (caddr sexpr))
      (define node (cmark_node_new type))
      (let ([gss (assq type getters+setters)])
        (when gss
          (unless (= (length (cdr gss)) (length info))
            (error 'sexpr->cmark "bad number of info values in ~s" sexpr))
          (for-each (λ(gs x) ((cdr gs) node x)) (cdr gss) info)))
      (cond [(string? data) (cmark_node_set_literal node data)]
            [(not data) (void)]
            [(list? data)
             (for ([child (in-list data)])
               (cmark_node_append_child node (sexpr->cmark child)))]
            [else (error 'sexpr->cmark "bad data in ~s" sexpr)])
      node)
    (define root (loop sexpr))
    (register-finalizer root cmark_node_free)
    root)

  ;; Registers a `cmark_node_free` finalizer
  (provide cmark-parse-document)
  (define (cmark-parse-document str [options '(normalize smart)])
    (define root (cmark_parse_document
                  (if (bytes? str) str (string->bytes/utf-8 str))
                  options))
    (register-finalizer root cmark_node_free)
    root)

  (provide cmark-render-html)
  (define (cmark-render-html root [options '(normalize smart)])
    (cmark_render_html root options)))

#; ;; sample use
(begin
  (require 'high-level racket/string)
  (cmark-render-html
   (cmark-parse-document
    (string-join '("foo"
                   "==="
                   ""
                   "> blah"
                   ">"
                   "> blah *blah* `bar()` blah:"
                   ">"
                   ">     function foo() {"
                   ">       bar();"
                   ">     }")
                 "\n"))))
