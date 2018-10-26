
#ifdef __cplusplus
extern "C" {
#endif

typedef int (*writer)(FILE*, void*);
typedef int (*bin_reader)(void*);
typedef int (*str_reader)(const char *, void*);

struct primtype_info_t {
  const char binname[4]; // Used for parsing binary data.
  const char* type_name; // Same name as in Futhark.
  const int size; // in bytes
  const writer write_str; // Write in text format.
  const str_reader read_str; // Read in text format.
  const writer write_bin; // Write in binary format.
  const bin_reader read_bin; // Read in binary format.
};

extern struct primtype_info_t i8_info;
extern struct primtype_info_t i16_info;
extern struct primtype_info_t i32_info;
extern struct primtype_info_t i64_info;
extern struct primtype_info_t u8_info;
extern struct primtype_info_t u16_info;
extern struct primtype_info_t u32_info;
extern struct primtype_info_t u64_info;
extern struct primtype_info_t f32_info;
extern struct primtype_info_t f64_info;
extern struct primtype_info_t bool_info;

int read_array(const struct primtype_info_t *expected_type, void **data,
    int64_t *shape, int64_t dims);
int write_array(FILE *out, int write_binary, const struct primtype_info_t
    *elem_type, void *data, int64_t *shape, int8_t rank);
int read_scalar(const struct primtype_info_t *expected_type, void *dest);
int write_scalar(FILE *out, int write_binary, const struct primtype_info_t
    *type, void *src);
#ifdef __cplusplus
}
#endif
